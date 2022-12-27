using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using Accord.Imaging.Filters;

using System.Drawing;
using NeuralNetwork1;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        private readonly BaseNetwork net1;
        private readonly BaseNetwork net2;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            net1 = new AccordNet(new int[] { 784, 300, 10 });
            net2 = new StudentNetwork(new int[] { 784, 392, 196, 98, 10 });
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                if (messageText == "train")
                {
                    if (!Directory.Exists(DataSet.PreparedPath))
                    {
                        Console.WriteLine("Preparing dataset...");
                        Directory.CreateDirectory(DataSet.PreparedPath);
                        DataSet.PrepareData();
                        Console.WriteLine("Done");
                    }
                    Console.WriteLine("Training on dataset...");
                    SamplesSet s = DataSet.GetDataSet();
                    net1.TrainOnDataSet(s, 10, 0.0005, true);
                    Console.WriteLine("Accurancy: " + net1.GetAccuracy(s));
                    net2.TrainOnDataSet(s, 50, 0.0005, true);
                    Console.WriteLine("Accurancy: " + net2.GetAccuracy(s));

                    Console.WriteLine("Training complete");
                    
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: "Обучение завершено",
                        cancellationToken: cancellationToken);
                }
                else
                {
                    var answer = aiml.Talk(chatId, username, messageText);
                    if (answer == "")
                        answer = "Не совсем вас понял";
                    // Echo received message text
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: answer,
                        cancellationToken: cancellationToken);
                }

                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");


                return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                // Если бы мы хотели получить битмап, то могли бы использовать new Bitmap(Image.FromStream(imageStream))
                // Но вместо этого пошлём картинку назад
                // Стрим помнит последнее место записи, мы же хотим теперь прочитать с самого начала

                var bmp = new System.Drawing.Bitmap(System.Drawing.Image.FromStream(imageStream));
                /* var filter = new ExtractBiggestBlob();
                 bmp = ImageHelper.MakeBW(bmp);
                 bmp = ImageHelper.Invert(bmp);
                 bmp = ImageHelper.CutSquare(bmp);
                 // bmp = ImageHelper.CutContent(bmp);
                 bmp = filter.Apply(bmp);
                 bmp = ImageHelper.Resize(bmp, new Size(28, 28));*/

                bmp = ImageHelper.PrepareBMP(bmp);
                var sample = new Sample(ImageHelper.GetArray(bmp), 10);
                var res = net1.Predict(sample);
                var res2 = net2.Predict(sample);
                Console.WriteLine("Accord result is " + res);
                Console.WriteLine("Student result is " + res2);
                //imageStream.Seek(0, 0);
                MemoryStream newImg = new MemoryStream();
                bmp.Save(newImg, System.Drawing.Imaging.ImageFormat.Jpeg);
                newImg.Seek(0, 0);

                await client.SendPhotoAsync(
                    message.Chat.Id,
                    newImg,
                    "Accord: Кажется, это " + res + '\n' +
                    "Student: Кажется, это " + res2,
                    cancellationToken: cancellationToken
                );
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }
    }
}
