using Accord.Imaging.Filters;
using NeuralNetwork1;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIMLTGBot
{
    public class ImageHelper
    {
        public static System.Drawing.Bitmap MakeBW(System.Drawing.Bitmap img)
        {
            using (Graphics gr = Graphics.FromImage(img)) // SourceImage is a Bitmap object
            {
                var gray_matrix = new float[][] {
                new float[] { 0.299f, 0.299f, 0.299f, 0, 0 },
                new float[] { 0.587f, 0.587f, 0.587f, 0, 0 },
                new float[] { 0.114f, 0.114f, 0.114f, 0, 0 },
                new float[] { 0,      0,      0,      1, 0 },
                new float[] { 0,      0,      0,      0, 1 }
            };

                var ia = new System.Drawing.Imaging.ImageAttributes();
                ia.SetColorMatrix(new System.Drawing.Imaging.ColorMatrix(gray_matrix));
                ia.SetThreshold(0.5f); // Change this threshold as needed
                var rc = new Rectangle(0, 0, img.Width, img.Height);
                gr.DrawImage(img, rc, 0, 0, img.Width, img.Height, GraphicsUnit.Pixel, ia);
            }

            return img;
        }

        public static System.Drawing.Bitmap Invert(System.Drawing.Bitmap img)
        {
            Bitmap pic = new Bitmap(img);
            for (int y = 0; (y <= (pic.Height - 1)); y++)
            {
                for (int x = 0; (x <= (pic.Width - 1)); x++)
                {
                    Color inv = pic.GetPixel(x, y);
                    inv = Color.FromArgb(255, (255 - inv.R), (255 - inv.G), (255 - inv.B));
                    pic.SetPixel(x, y, inv);
                }
            }
            return pic;
        }

        public static System.Drawing.Bitmap CutContent(System.Drawing.Bitmap img)
        {
            System.Drawing.Bitmap pic = new Bitmap(img);
            int left_bound = img.Width;
            int right_bound = 0;
            int top = 0;
            int bottom = img.Height;

            bool first = true;

            for (int y = 0; (y <= (pic.Height - 1)); y++)
            {
                for (int x = 0; (x <= (pic.Width - 1)); x++)
                {
                    var color = pic.GetPixel(x, y);
                    if ((255 - color.R) < 10)
                    {
                        if (first)
                        {
                            left_bound = x;
                            top = y;
                            first = false;
                        }
                        else
                        {
                            if(x > left_bound && x>right_bound)
                                right_bound = x;
                            bottom = y;
                        }
                    }
                }
            }

            right_bound += 40;
            left_bound = 0;
            top -= 40;
            bottom += 40;
            var width = right_bound - left_bound;
            var height = bottom - top;
            var destImage = new Bitmap(width, height);
            using (var g = Graphics.FromImage(destImage))
            {
                using (System.Drawing.Imaging.ImageAttributes attributes = new System.Drawing.Imaging.ImageAttributes())
                {
                    Rectangle section = new Rectangle(new Point(0, 0), new Size(width, height));
                    g.DrawImage(img, section, left_bound, top, width, height, GraphicsUnit.Pixel, attributes);
                }
            }

            return destImage;

        }

        public static System.Drawing.Bitmap CutSquare(System.Drawing.Bitmap img)
        {
            int left_x = 0, bottom_y = 0, length = img.Width;
            if (img.Height > img.Width)
            {
                length = img.Width;
                bottom_y = (img.Height - length) / 2;
            }
            else if (img.Height < img.Width)
            {
                length = img.Height;
                left_x = (img.Width - length) / 2;
            }
            var destImage = new Bitmap(length, length);
            using (var g = Graphics.FromImage(destImage))
            {
                using (System.Drawing.Imaging.ImageAttributes attributes = new System.Drawing.Imaging.ImageAttributes())
                {
                    Rectangle section = new Rectangle(new Point(0, 0), new Size(length, length));
                    g.DrawImage(img, section, left_x, bottom_y, length, length, GraphicsUnit.Pixel, attributes);
                }
            }

            return destImage;
        }

        public static Bitmap Resize(Image image, Size size)
        {
            var destRect = new Rectangle(0, 0, size.Width, size.Height);
            var destImage = new Bitmap(size.Width, size.Height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new System.Drawing.Imaging.ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static double[] GetArray(Bitmap pic)
        {
            var result = new double[pic.Height * pic.Width];
            for (int j = 0; j < pic.Height; ++j)
            {
                for (int i = 0; i < pic.Width; ++i)
                {
                    result[j * pic.Width + i] = (pic.GetPixel(i, j).R) > 127 ? 1 : 0;
                }
            }
            return result;
/*            var res = new double[56];
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (pic.GetPixel(i, j).R > 127)
                    {
                        res[i]++;
                    }
                }
            }
            for (int i = 28; i < 56; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (pic.GetPixel(j, i % 28).R > 127)
                    {
                        res[i]++;
                    }
                }
            }
            return res;*/
        }

        public static Bitmap PrepareBMP(Bitmap bmp)
        {
            var filter = new ExtractBiggestBlob();
            bmp = MakeBW(bmp);
            bmp = Invert(bmp);
            bmp = CutSquare(bmp);
            // bmp = ImageHelper.CutContent(bmp);
            bmp = filter.Apply(bmp);
            bmp = Resize(bmp, new Size(28, 28));
            return bmp;
        }
    }
}
