using Avalonia.Controls;
using Avalonia.Interactivity;

namespace _04_Click_Detection_App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void OnButtonClick(object sender, RoutedEventArgs e)
    {
        outputTextBox.Text = "Button was clicked!";
     }
}