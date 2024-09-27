#pragma once

namespace ml
{

class DenseLayer
{
public:

    // Default constructor deleted.
    DenseLayer() = delete;

    /**
     * @brief Creates a new dense layer.
     * 
     * @param nodeCount   The number of nodes in the new layer.
     * @param weightCount The number of weights per node in the layer.
     */
    DenseLayer(const std::size_t nodeCount, const std::size_t weightCount);

    /**
     * @brief Provides the number of nodes in the dense layer.
     * 
     * @return The number of nodes in the layer as an integer.
     */
    std::size_t nodeCount() const;

    /**
     * @brief Provides the number of weights per node in the dense layer.
     * 
     * @return The number of weights per node in the layer as an integer.
     */
    std::size_t weightCount() const;

    /**
     * @brief Provides the bias value of each node.
     * 
     * @return Reference to a vector holding the bias of each node.
     */
    const std::vector<double>& bias() const;

    // Skapa get-metoder för att läsa outputs, errors och vikter.

private:
    std::vector<double> myOutput;               // Vector holding the output of each node.
    std::vector<double> myError;                // Vector holding the error of each node.
    std::vector<double> myBias;                 // Vector holding the bias of each node.
    std::vector<std::vector<double>> myWeights; // Vector holding the weights of each node.

};

} // namespace ml.

// Lägg till get- och set-metoder om ni finner det lämpligt.

// Fundera vilka publika metoder som behövs. Vi behöver kunna köra.
// 1. Feedforward med en ny input (förslagsvis i en vektor).
// 2. Backpropagera med referensdata (förslagsvis  i en vektor).
// 3. Backpropagera med nästa lager (om ett sådant finns).
// 4. Optimera våra parametrar (förslags med input i en vektor).