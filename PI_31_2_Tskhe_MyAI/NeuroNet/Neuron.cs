using System;
using System.Drawing.Text;
using static System.Math;

namespace PI_31_2_Tskhe_MyAI.NeuroNet
{
    class Neuron
    {
        // поля 
        private NeuronType type; // тип нейрона
        private double[] weights; // веса
        private double[] inputs; //входы
        private double output; // выход
        private double derivative; // производная 

        // константы для функции активации (утекающая релу)
        // private double a = 0.01d; 

        //private double a = 1;
        //private double b = 1;

        // свойства 
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get  => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // конструктор 
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }
        
        // метод активации нейрона (нелинейные преобразования входного сигнала)
        public void Activator(double[] i)
        {
            inputs = i; //передача вектора входного сигнала в массив входных данных нейрона

            double sum = weights[0]; // аффиное преобразование через смещение

            for (int j = 0; j < inputs.Length; j++) //цикл вычисления индуцированного поля
            {
                sum += inputs[j] * weights[j + 1]; //линейные преобразовавния входных сигналов
            }

            switch (type)
            {
                case NeuronType.Hidden:
                    output = TanhFunc(sum); //заменить на свою функцию активации
                    derivative = TanhDerivative(sum);
                    break;
                case NeuronType.Output:
                    output = Exp(sum);
                    break;
            }

        }
        //функция активации нейрона (свою написать)
        private double TanhFunc(double sum)
        {
            if (sum >= 20) return 1;
            if (sum <= -20) return -1;
            // return (Exp(2 * sum * a) - 1) / (Exp(2 * sum * b) + 1);
            return Tanh(sum);
        }
        private double TanhDerivative(double sum)
        {
            double tanh = TanhFunc(sum);
            return 1 - tanh * tanh;
        }
    }
}
