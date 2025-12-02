using System;
using System.Linq;

namespace PI_31_2_Tskhe_MyAI.NeuroNet
{
    class Network
    {
        // все слои сети
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(36, 71, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 36, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; // массив фактического выхода сети
        private double[] e_error_avr; // среднее значение энергии ошибки эпохи обучения
        private double[] train_accuracy;
        private double[] test_accuracy;

        // свойства
        public double[] Fact { get => fact; } // массив фактического выхода сети

        // среднее значение энергии ошибки эпохи обучения
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }
        public double[] Train_accuracy { get => train_accuracy; set => train_accuracy = value; }
        public double[] Test_accuracy { get => test_accuracy; set => test_accuracy = value; }


        // конструктор
        public Network() { }



        // прямой проход сети
        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }


        public void Train(Network net) //backpropagation method
        {
            net.input_layer = new InputLayer(NetworkMode.Train);
            int epoches = 10; // кол-во эпох обучения
            double tmpSumError;     // временная переменная суммы ошибок
            double[] errors;        // вектор сигнала ошибки выходного слоя
            double[] temp_gsums1;   // вектор градиента 1-го скрытого слоя
            double[] temp_gsums2;   // вектор градиента 2-го скрытого слоя


            e_error_avr = new double[epoches];
            train_accuracy = new double[epoches];


            for (int k = 0; k < epoches; k++) // перебор эпох обучения
            {
                e_error_avr[k] = 0; // значение средней ошибки

                int correct_predictions = 0;

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset);
                for (int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    ForwardPass(net, tmpTrain); // прямой проход обучающего образа

                    // вычисление ошибки по итерации
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0]) errors[x] = 1.0 - net.fact[x];
                        else errors[x] = -net.fact[x]; // 0.0 - net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // суммарное значение энергии ошибки

                    int predicted = net.fact.ToList().IndexOf(net.fact.Max());
                    int actual = (int)net.input_layer.Trainset[i, 0];
                    if (predicted == actual) correct_predictions++;

                    // обратный проход и коррекция весов 
                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0); // среднее значение энергии ошибки
                train_accuracy[k] = correct_predictions / net.input_layer.Trainset.GetLength(0);
            }

            net.input_layer = null; //обнуление (уборка) входного слоя

            // запись скорректированных весов в "память"
            net.hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            net.output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }
        public void Test(Network net) 
        {
            net.input_layer = new InputLayer(NetworkMode.Test);
            int epoches = 5; // кол-во эпох тестирования

            double tmpSumError;     // временная переменная суммы ошибок
            double[] errors;        // вектор сигнала ошибки выходного слоя
            e_error_avr = new double[epoches];

            test_accuracy = new double[epoches];

            for (int k = 0; k < epoches; k++) // перебор эпох тестирования
            {
                e_error_avr[k] = 0; // значение средней ошибки
                int correct_predictions = 0;

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Testset);
                for (int i = 0; i < net.input_layer.Testset.GetLength(0); i++)
                { 
                    double[] tmpTest = new double[15];
                    for (int j = 0; j < tmpTest.Length; j++)
                        tmpTest[j] = net.input_layer.Testset[i, j + 1];

                    ForwardPass(net, tmpTest); // прямой проход тестового образа

                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Testset[i, 0]) errors[x] = 1.0 - net.fact[x];
                        else errors[x] = -net.fact[x]; // 0.0 - net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    int predicted = net.fact.ToList().IndexOf(net.fact.Max());
                    int actual = (int)net.input_layer.Testset[i, 0];
                    if (predicted == actual) correct_predictions++;
                    
                    e_error_avr[k] += tmpSumError / errors.Length; // суммарное значение энергии ошибки
                }
                test_accuracy[k] = correct_predictions / net.input_layer.Testset.GetLength(0);
            }

            net.input_layer = null; //обнуление (уборка) входного слоя
        }

        public void Dropout(Network net)
        {
            Random rand = new Random();

            net.hidden_layer1.DropoutToLayer(rand);
            net.hidden_layer2.DropoutToLayer(rand);
            net.output_layer.DropoutToLayer(rand);
        }
    }
}
