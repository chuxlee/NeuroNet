using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using PI_31_2_Tskhe_MyAI.NeuroNet;

namespace PI_31_2_Tskhe_MyAI
{
    public partial class FormMain : Form
    {
        // HiddenLayer hiddenLayer; //DELETE IT LATER
        private double[] inputPixels; // массив входных данных 
        private Network network; // объявление нейросети
        public FormMain()
        {
            InitializeComponent();

            inputPixels = new double[15];
            network = new Network();
        }

        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            if (((Button)sender).BackColor == Color.White)
            {
                ((Button)sender).BackColor = Color.Black;
                inputPixels[((Button)sender).TabIndex] = 1d;
            }
            else
            {
                ((Button)sender).BackColor = Color.White;
                inputPixels[((Button)sender).TabIndex] = 0d;
            }
        }

        
        private void SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void buttonRecognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            label_output.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            label_Probability.Text = (100 * network.Fact.Max()).ToString("0.00") + "%";
        }

        private void button_Training_Click(object sender, EventArgs e)
        {
            network.Train(network);
            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
            }
            MessageBox.Show("Обучение успешно завершено.", "Информация",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void SaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr);
        }

        private void button_test_Click(object sender, EventArgs e)
        {
            network.Test(network);
            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
            }
            MessageBox.Show("Тестирование завершено.", "Информация",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void button_dropout_Click(object sender, EventArgs e)
        {
            network.Dropout(network);
            MessageBox.Show("Регуляризация завершена.", "Информация",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
