use rand::Rng;

// Define activation functions
#[derive(Clone)] // Derive Clone trait for ActivationFunction
enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
}

impl ActivationFunction {
    fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()), // Sigmoid activation function
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 }, // ReLU activation function
            ActivationFunction::Tanh => x.tanh(), // Hyperbolic tangent activation function
        }
    }
}

// Define a basic neuron structure
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    activation_fn: ActivationFunction,
}

impl Neuron {
    fn new(input_size: usize, activation_fn: ActivationFunction) -> Self {
        // Initialize weights randomly and set bias
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size).map(|_| rng.gen::<f64>()).collect();
        let bias = rng.gen::<f64>();
        
        Neuron { weights, bias, activation_fn }
    }
    
    fn activate(&self, inputs: &[f64]) -> f64 {
        // Perform weighted sum of inputs and apply activation function
        let weighted_sum: f64 = inputs.iter().zip(&self.weights).map(|(&x, &w)| x * w).sum();
        (self.activation_fn).activate(weighted_sum + self.bias)
    }
}

// Define a simple neural network layer
struct NeuralLayer {
    neurons: Vec<Neuron>,
}

impl NeuralLayer {
    fn new(input_size: usize, num_neurons: usize, activation_fn: ActivationFunction) -> Self {
        let neurons: Vec<Neuron> = (0..num_neurons).map(|_| Neuron::new(input_size, activation_fn.clone())).collect();
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
    fn new(layer_sizes: &[usize], activation_fns: Vec<ActivationFunction>) -> Self {
        assert_eq!(layer_sizes.len() - 1, activation_fns.len(), "Number of activation functions must match number of layers");

        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            let input_size = layer_sizes[i - 1];
            let num_neurons = layer_sizes[i];
            let activation_fn = activation_fns[i - 1].clone();
            layers.push(NeuralLayer::new(input_size, num_neurons, activation_fn));
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
    let activation_fns = vec![
        ActivationFunction::Sigmoid,
        ActivationFunction::Sigmoid,
    ];
    let network = NeuralNetwork::new(&[2, 3, 1], activation_fns); // Define a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    let inputs = vec![0.5, 0.3];
    let outputs = network.forward(&inputs);
    println!("Output: {:?}", outputs);
}
