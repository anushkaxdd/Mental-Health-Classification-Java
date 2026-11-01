
import ai.onnxruntime.*;
import java.util.*;
import java.nio.LongBuffer;

public class Main {

    private static final String MODEL_PATH =
        "/content/MentalHealthBERT/models/model.onnx";

    private static final String VOCAB_PATH =
        "/content/MentalHealthBERT/models/vocab.txt";

    private static final int MAX_LENGTH = 128;

    private static final String[] LABELS = {
        "Anxiety", "Depression", "Normal", "Suicidal"
    };

    public static void main(String[] args) throws Exception {

        String text =
            "I've been feeling really overwhelmed and anxious lately.";

        System.out.println("MENTAL HEALTH BERT - JAVA INFERENCE");
        System.out.println("-----------------------------------");

        // Load tokenizer
        Tokenizer tokenizer = new Tokenizer(VOCAB_PATH);

        // Convert text into token IDs
        int[] ids = tokenizer.encode(text);
        int[] mask = tokenizer.attentionMask(ids);
        int[] segments = tokenizer.tokenTypeIds();

        // Convert integer arrays to long arrays
        long[] inputIds = toLongArray(ids);
        long[] attentionMask = toLongArray(mask);
        long[] tokenTypes = toLongArray(segments);

        // Load ONNX model
        OrtEnvironment environment = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options =
            new OrtSession.SessionOptions();

        OrtSession session =
            environment.createSession(MODEL_PATH, options);

        // Create input tensors
        long[] shape = {1, MAX_LENGTH};

        OnnxTensor idsTensor = OnnxTensor.createTensor(
            environment, LongBuffer.wrap(inputIds), shape
        );

        OnnxTensor maskTensor = OnnxTensor.createTensor(
            environment, LongBuffer.wrap(attentionMask), shape
        );

        OnnxTensor typesTensor = OnnxTensor.createTensor(
            environment, LongBuffer.wrap(tokenTypes), shape
        );

        Map<String, OnnxTensor> inputs = new HashMap<>();

        inputs.put("input_ids", idsTensor);
        inputs.put("attention_mask", maskTensor);
        inputs.put("token_type_ids", typesTensor);

        // Run BERT
        OrtSession.Result result = session.run(inputs);

        float[][] logits =
            (float[][]) result.get("logits").get().getValue();

        // Calculate softmax probabilities
        double[] probabilities = softmax(logits[0]);

        // Find predicted class
        int predictedClass = 0;

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[predictedClass]) {
                predictedClass = i;
            }
        }

        System.out.println("\nInput text:");
        System.out.println(text);

        System.out.println("\nClassification probabilities:");

        for (int i = 0; i < LABELS.length; i++) {
            System.out.printf(
                "%-12s : %.4f (%.2f%%)%n",
                LABELS[i],
                probabilities[i],
                probabilities[i] * 100
            );
        }

        System.out.println("\nPredicted classification: "
            + LABELS[predictedClass]);

        System.out.printf(
            "Confidence: %.2f%%%n",
            probabilities[predictedClass] * 100
        );

        // Close resources
        result.close();
        idsTensor.close();
        maskTensor.close();
        typesTensor.close();
        session.close();
        options.close();

        System.out.println("\nJava BERT inference completed!");
    }

    private static long[] toLongArray(int[] values) {

        long[] output = new long[values.length];

        for (int i = 0; i < values.length; i++) {
            output[i] = values[i];
        }

        return output;
    }

    private static double[] softmax(float[] logits) {

        double max = Arrays.stream(toDoubleArray(logits))
            .max().orElse(0);

        double sum = 0;
        double[] probabilities = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - max);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    private static double[] toDoubleArray(float[] values) {

        double[] output = new double[values.length];

        for (int i = 0; i < values.length; i++) {
            output[i] = values[i];
        }

        return output;
    }
}
