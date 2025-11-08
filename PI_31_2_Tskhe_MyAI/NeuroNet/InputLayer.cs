using System;
using System.Collections.Specialized;
using System.Drawing.Printing;
using System.IO;

namespace PI_31_2_Tskhe_MyAI.NeuroNet
{
    class InputLayer
    {
        private double[,] trainset; // 100 изображений в обучающей выборке
        private double[,] testset; // 10 изображений в тестовой выборке

        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory;
            string[] tmpArrStr; // временный массив строк
            string[] tmpStr;    // временный массив элементов в строке

            switch(nm)
            {
                case NetworkMode.Train:
                    tmpArrStr = File.ReadAllLines(path + "train.txt"); 
                    trainset = new double[tmpArrStr.Length, 16];

                    for (int i = 0; i < tmpArrStr.Length; i++)
                    {
                        tmpStr = tmpArrStr[i].Split(' ');

                        for (int j = 0; j < 16; j++)
                        {
                            trainset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }
                    Shuffling_Array_Rows(trainset); // перетасовка методом Фишера-Йетса 
                    break;
                case NetworkMode.Test:
                    tmpArrStr = File.ReadAllLines(path + "test.txt");
                    testset = new double[tmpArrStr.Length, 16];

                    for (int i = 0; i < tmpArrStr.Length; i++)
                    {
                        tmpStr = tmpArrStr[i].Split(' ');

                        for (int j = 0; j < 16; j++)
                        {
                            testset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }
                    Shuffling_Array_Rows(testset); // перетасовка методом Фишера-Йетса 
                    break;
            }
        }

        public void Shuffling_Array_Rows(double[,] arr)
        {
            int rowCount = arr.GetLength(0);
            int colCount = arr.GetLength(1);
            Random random = new Random();

            for (int i = rowCount - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                SwapRows(arr, i, j, colCount);
            }
        }

        private void SwapRows(double[,] arr, int row1, int row2, int colCount)
        {
            for (int col = 0; col < colCount; col++)
            { 
                double temp = arr[row1, col];
                arr[row1, col] = arr[row2, col];
                arr[row2, col] = temp;
            }
        }
    }
}
