using Avalonia.Controls;
using Avalonia.Interactivity;

namespace _04_Click_Detection_App
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void OnStartButtonClick(object sender, RoutedEventArgs e)
        {
            StartRecording();
        }

        private void OnStopButtonClick(object sender, RoutedEventArgs e)
        {
            StopRecording();
        }

        private void StartRecording()
        {
            AmplitudeTextBox.Text = "Recording started";
        }

        private void StopRecording()
        {
            AmplitudeTextBox.Text = "Recording stopped";
        }
    }
}
