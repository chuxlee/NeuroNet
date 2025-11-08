using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PI_31_2_Tskhe_MyAI.NeuroNet;

namespace PI_31_2_Tskhe_MyAI.NeuroNet
{
    internal class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string type) : 
            base(non, nopn, nt, type)
        { }

        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
                hidden_out[i] = neurons[i].Output;
            nextLayer.Data = hidden_out; // передача выходного сигнала на вход следующего слоя 
        }

        //Обратный проход
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];
            
            for (int j = 0; j < numofprevneurons; j++) //цикл вычисления градиентной суммы j-го нейрона
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                    sum += neurons[k].Weights[j] * neurons[k].Derivative * gr_sums[k]; //через градиентные суммы и производную

                gr_sum[j] = sum;
            }

            for (int i = 0; i < numofneurons; i++) //цикл коррекции синаптических весов 
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0)
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * neurons[i].Derivative * gr_sums[i];
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * gr_sums[i];

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw; // коррекция весов 
                }

            return gr_sum;
        }
    }
}
