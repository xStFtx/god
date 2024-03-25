use rand::Rng;
// Define a basic neuron structure
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(input_size: usize) -> Self {
        // Initialize weights randomly and set bias
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size).map(|_| rng.gen::<f64>()).collect();
        let bias = rng.gen::<f64>();
        
        Neuron { weights, bias }
    }
    
    fn activate(&self, inputs: &[f64]) -> f64 {
        // Perform weighted sum of inputs and apply activation function
        let weighted_sum: f64 = inputs.iter().zip(&self.weights).map(|(&x, &w)| x * w).sum();
        1.0 / (1.0 + (-weighted_sum - self.bias).exp()) // Sigmoid activation function
    }
}

// Define a simple neural network layer
struct NeuralLayer {
    neurons: Vec<Neuron>,
}

impl NeuralLayer {
    fn new(input_size: usize, num_neurons: usize) -> Self {
        let neurons: Vec<Neuron> = (0..num_neurons).map(|_| Neuron::new(input_size)).collect();
        NeuralLayer { neurons }
    }
    
    fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.activate(inputs)).collect()
    }
}

// Define a simple neural network
struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            let input_size = layer_sizes[i - 1];
            let num_neurons = layer_sizes[i];
            layers.push(NeuralLayer::new(input_size, num_neurons));
        }
        NeuralNetwork { layers }
    }
    
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut activations = inputs.to_vec();
        for layer in &self.layers {
            activations = layer.activate(&activations);
        }
        activations
    }
}

fn main() {
    // Example usage
    let network = NeuralNetwork::new(&[2, 3, 1]); // Define a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    let inputs = vec![0.5, 0.3];
    let outputs = network.forward(&inputs);
    println!("Output: {:?}", outputs);
}
