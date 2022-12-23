using System;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private Random rand;
        bool parallel = false;
        private double[][,] weights;
        private double[][] layers;
        private double[][] errors;
        public double learning_rate = 0.05;

        private double Sigmoid(double x)
        {
            return 1f / (1f + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1f - x);
        }

        /// <summary>
        /// Randomize weghts of the network
        /// </summary>
        private void RandomizeWeights()
        {
            if (parallel)
            {
                Parallel.ForEach(weights, w => // Parallel for each weight matrix
                {
                    for (int i = 0; i < w.GetLength(0); i++)  // For each row
                    {
                        for (int j = 0; j < w.GetLength(1); j++) // For each column
                        {
                            w[i, j] = rand.NextDouble() * 2 - 1; // Randomize
                        }
                    }
                });
            }
            else
            {
                foreach (var w in weights) // For each weight matrix
                {
                    for (int i = 0; i < w.GetLength(0); i++)  // For each row
                    {
                        for (int j = 0; j < w.GetLength(1); j++) // For each column
                        {
                            w[i, j] = rand.NextDouble() * 2 - 1; // Randomize
                        }
                    }
                }
            }
        }

        private void ForwardPass()
        {
            if (parallel)
            {
                for (int k = 1; k < layers.Length; k++) // For each layer
                {
                    Parallel.For(0, weights[k - 1].GetLength(1), j => // For each neuron in the layer
                    {
                        double sum = 0;
                        for (int i = 0; i < weights[k - 1].GetLength(0); i++) // For each weight
                        {
                            sum += weights[k - 1][i, j] * layers[k - 1][i]; // Multiply weight by input
                        }
                        layers[k][j] = Sigmoid(sum); // Apply sigmoid function
                    });
                }
            }
            else
            {
                for (int k = 1; k < layers.Length; k++) // For each layer
                {
                    for (int j = 0; j < weights[k - 1].GetLength(1); j++) // For each neuron in the layer
                    {
                        double sum = 0;
                        for (int i = 0; i < weights[k - 1].GetLength(0); i++) // For each weight
                        {
                            sum += weights[k - 1][i, j] * layers[k - 1][i]; // Multiply weight by input
                        }
                        layers[k][j] = Sigmoid(sum); // Apply sigmoid function
                    }
                }
            }
        }

        /// <param name="ans_idx">index of correct answer in output vector</param>
        private void BackPropogation(int ans_idx)
        {
            if (parallel)
            {
                int k = layers.Length - 1; // last layer
                Parallel.For(0, layers[k].Length, j => // For each neuron in last layer
                {
                    double n = layers[k][j];
                    errors[k][j] = -SigmoidDerivative(n) * ((j == ans_idx ? 1f : 0f) - n); // calculate error
                });

                // inner layers
                for (k = layers.Length - 2; k > 0; k--) // For each inner layer
                {
                    Parallel.For(0, layers[k].Length - 1, i => // For each neron in layer
                    {
                        errors[k][i] = 0; // reset error
                        for (int j = 0; j < weights[k].GetLength(1); j++) // For each weight on same layer
                        {
                            errors[k][i] += weights[k][i, j] * errors[k + 1][j]; // Accumulate weight multiplied by error
                        }
                        errors[k][i] *= SigmoidDerivative(layers[k][i]); // Apply sigmoid derivative
                    });
                }

                // update weights
                for (k = 0; k < weights.Length; k++)
                {
                    Parallel.For(0, weights[k].GetLength(0), i =>
                    {
                        for (int j = 0; j < weights[k].GetLength(1); j++)
                        {
                            // Update weights according to learning rate
                            weights[k][i, j] += -learning_rate * errors[k + 1][j] * layers[k][i];
                        }
                    });
                }
            }
            else
            {
                int k = layers.Length - 1; // last layer
                for (int j = 0; j < layers[k].Length; j++) // For each neuron in last layer
                {
                    double n = layers[k][j];
                    errors[k][j] = -SigmoidDerivative(n) * ((j == ans_idx ? 1f : 0f) - n); // calculate error
                }

                // inner layers
                for (k = layers.Length - 2; k > 0; k--)  // For each inner layer
                {
                    for (int i = 0; i < layers[k].Length - 1; i++) // For each neron in layer
                    {
                        for (int j = 0; j < weights[k].GetLength(1); j++) // For each weight on same layer
                        {
                            errors[k][i] += weights[k][i, j] * errors[k + 1][j]; // Accumulate weight multiplied by error
                        }
                        errors[k][i] *= SigmoidDerivative(layers[k][i]);  // Apply sigmoid derivative
                    }
                }

                // update weights
                for (k = 0; k < weights.Length; k++)
                {
                    for (int i = 0; i < weights[k].GetLength(0); i++)
                    {
                        for (int j = 0; j < weights[k].GetLength(1); j++)
                        {
                            // Update weights according to learning rate
                            weights[k][i, j] += -learning_rate * errors[k + 1][j] * layers[k][i];
                        }
                    }
                }
            }
        }

        public StudentNetwork(int[] structure)
        {
            /*
            [ x1 ]  [ w11 ... w1m ]      [ l1 ]  [ w11 ... w1m ]      [ y1 ]
            [ x2 ]  [ w21 ... w2m ]      [ l2 ]  [ w21 ... w2m ]      [ y2 ]
            [ .. ]  [ ... ... ... ]  ->  [ .. ]  [ ... ... ... ]  ->  [ .. ]
            [ xn ]  [ wn1 ... wnm ]      [ ln ]  [ wn1 ... wnm ]      [ yn ]
            [ 1  ]                       [ 1  ]                      
            (bias)                       (bias)                      (no bias)
            in layer                     hidden layer                out layer
            */

            rand = new Random();
            layers = new double[structure.Length][];
            layers[0] = new double[structure[0] + 1]; // +1 for bias
            layers[0][structure[0]] = 1; // bias
            errors = new double[structure.Length][];
            weights = new double[structure.Length - 1][,]; // -1 for output layer
            for (int i = 1; i < structure.Length; i++)
            {
                if (i == structure.Length - 1) // output layer
                    layers[i] = new double[structure[i]]; // no bias
                else // not input layer
                {
                    layers[i] = new double[structure[i] + 1]; // +1 for bias
                    layers[i][structure[i]] = 1; // bias
                }
                errors[i] = new double[structure[i]];
                if (i != structure.Length - 1) // not output layer
                    layers[i][structure[i]] = 1; // bias
                weights[i - 1] = new double[structure[i - 1] + 1, structure[i]]; // +1 for bias
            }
            RandomizeWeights();
            LoadFromDisk();
        }

        ~StudentNetwork()
        {
            SaveToDisk();
        }

        bool LoadFromDisk(string path = "weights.csv")
        {
            // Load weights from disk
            if (File.Exists(path))
            {
                StreamReader f = File.OpenText(path);
                int layers = int.Parse(f.ReadLine());
                for (int i = 0; i < layers; i++)
                {
                    int rows = int.Parse(f.ReadLine());
                    int cols = int.Parse(f.ReadLine());
                    for (int j = 0; j < rows; j++)
                    {
                        string[] line = f.ReadLine().Split(';');
                        for (int k = 0; k < cols; k++)
                        {
                            weights[i][j, k] = double.Parse(line[k]);
                        }
                    }
                }
                f.Close();
            }
            return false;
        }

        bool SaveToDisk(string path = "weights.csv")
        {
            // Save weights to disk
            StreamWriter f = File.CreateText(path);
            f.WriteLine(weights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                f.WriteLine(weights[i].GetLength(0));
                f.WriteLine(weights[i].GetLength(1));
                for (int j = 0; j < weights[i].GetLength(0); j++)
                {
                    for (int k = 0; k < weights[i].GetLength(1); k++)
                    {
                        f.Write(weights[i][j, k]);
                        if (k != weights[i].GetLength(1) - 1)
                            f.Write(';');
                    }
                    f.WriteLine();
                }
            }
            f.Close();
            return true;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            this.parallel = parallel;
            double est_error = 0; // estimated error
            int epochs = 0; // number of epochs


            while (true)
            {
                if (Predict(sample) == sample.actualClass && // if predicted class is correct
                    (est_error = sample.EstimatedError()) < acceptableError) // and estimated error is less than acceptable
                {
                    return epochs; // return number of epochs
                }
                else
                {
                    epochs++; // increment number of epochs
                    BackPropogation((int)sample.actualClass); // backpropogate
                }
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            this.parallel = parallel;
            double est_error = 0; // estimated error

            for (int epoch = 0; epoch < epochsCount; epoch++) // for each epoch
            {
                est_error = 0; // reset estimated error
                foreach (var sample in samplesSet.samples) // for sample in smaples set
                {
                    Predict(sample); // predict to get estimated error
                    est_error += sample.EstimatedError(); // accumulate estimated error
                    BackPropogation((int)sample.actualClass); // backpropogate
                }
                double error = est_error / samplesSet.samples.Count; // average error
                if (error < acceptableError) // if error is less than acceptable
                {
                    return error; // return error
                }
            }
            return est_error / samplesSet.Count; // return average error
        }

        /// <summary>
        /// Calculates the output of the network for the given input
        /// </summary>
        /// <param name="input"></param>
        protected override double[] Compute(double[] input)
        {
            if (parallel)
            {
                Parallel.For(0, input.Length, i =>
                {
                    layers[0][i] = input[i]; // set input
                });
            }
            else
            {
                for (int i = 0; i < input.Length; i++)
                {
                    layers[0][i] = input[i]; // set input
                }
            }
            ForwardPass(); // forward pass
            return layers[layers.Length - 1]; // return output
        }
    }
}
