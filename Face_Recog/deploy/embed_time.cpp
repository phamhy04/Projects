
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>
using namespace std;

int main()
{
    // Load script model
    torch::jit::script::Module model = torch::jit::load("../models/script_model.pt", torch::kCUDA);
    // Create faces for time testing
    vector<torch::jit::IValue> inputs;
    inputs.emplace_back(torch::randn({ 20, 3, 224, 224 }).to(at::kCUDA));
    float time = 0;
    for(int i=0; i<20; i++)
    {   
        auto start = chrono::steady_clock::now();
        auto outputs = model.forward(inputs).toTensor();
        auto end = chrono::steady_clock::now();
        // Calculate elaps time
        double elaps_time = double(chrono::duration_cast <chrono::nanoseconds> (end - start).count())/1e9;
        cout << "Elaps time (s): " << elaps_time << endl;
        time += elaps_time;
    }
    cout << "=>=> Everage time (s): " << time/20  << endl;;    
    return 0;
}