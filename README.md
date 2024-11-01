# **TEXT-GENERATOR**

## Overview

TEXT-GENERATOR is a natural language generation application designed to create high-quality text based on custom prompts. The model integrates Bidirectional Long Short-Term Memory (Bi-LSTM) and LSTM networks with a pre-trained large language model, using PyTorch for robust training and inference. Additionally, attention mechanisms and prompt engineering are employed to refine the quality of generated text, making it more contextually relevant and coherent.

The application includes an interactive user interface built with Streamlit, allowing users to generate text, experiment with prompt variations, and evaluate output quality using BLEU and ROUGE metrics.

## Features

- **Bi-LSTM and LSTM Model Architecture**: Uses Bi-LSTM and LSTM layers for improved contextual understanding, combined with a pre-trained language model.
- **Attention Mechanisms**: Focuses on relevant parts of the input sequence, improving text quality.
- **Interactive Streamlit Interface**: Allows real-time experimentation with prompts and customizations.
- **Evaluation Metrics**: Integrated BLEU and ROUGE metrics to assess text quality in terms of coherence and relevance.
- **Scalable and Modular Codebase**: Designed for easy modifications and enhancements.

## Sample Output

Example prompt: **"The future of AI in healthcare"**

**Generated Text**:
> "The future of AI in healthcare holds immense promise. With advancements in machine learning and deep learning, AI is expected to revolutionize diagnostics, treatment plans, and patient care. By analyzing large volumes of data, AI can uncover insights that were previously inaccessible, enhancing precision in personalized treatments and preventive care. Additionally, AI-driven automation is anticipated to reduce administrative burdens, allowing healthcare providers to focus more on patient care."

### Evaluation

If a reference text is provided, the application can evaluate the output text using BLEU and ROUGE scores. Here’s an example based on a sample reference:

- **Reference Text**: "AI will transform healthcare by enhancing diagnostics, personalized treatment, and operational efficiency, allowing more focus on patient-centered care."
- **Generated Text Evaluation**:
  - **BLEU Score**: 0.75
  - **ROUGE Scores**:
    - ROUGE-1: 0.8
    - ROUGE-2: 0.65
    - ROUGE-L: 0.78

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/TEXT-GENERATOR.git
   cd TEXT-GENERATOR
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Model**:
   - Enter your prompt in the input box.
   - Click **Generate Text** to see the output.
   - Optionally, enter a reference text to evaluate the generated text using BLEU and ROUGE metrics.

## Project Structure

```
TEXT-GENERATOR/
├── model/
│   ├── text_generator.py      # Model architecture and text generation
│   ├── attention.py           # Attention mechanism
│   └── utils.py               # Utility functions for tokenization and evaluation
├── app.py                     # Main Streamlit application
└── requirements.txt           # Dependencies
```

## Future Enhancements

- **Enhanced Attention Mechanisms**: Add advanced attention models for more detailed understanding of input sequences.
- **Transformer Integration**: Experiment with transformer-based models for higher-quality text generation.
- **Enhanced Interface**: Allow users to save custom prompts and outputs, and add parameter adjustments for further customization.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [NLTK](https://www.nltk.org/)

