import ai.onnxruntime.*;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Evaluation {

    static final String[] LABELS = {
        "Anxiety", "Depression", "Normal", "Suicidal"
    };

    static final String MODEL_PATH = "models/model.onnx";
    static final String VOCAB_PATH = "models/vocab.txt";
    static final String TEST_PATH = "models/test.csv";
    static final String RESULTS_PATH = "results";

    static Tokenizer tokenizer;
    static OrtEnvironment env;
    static OrtSession session;

    static class Prediction {
        int predicted;
        double[] probabilities;

        Prediction(int predicted, double[] probabilities) {
            this.predicted = predicted;
            this.probabilities = probabilities;
        }
    }

    static Prediction predict(String text) throws Exception {

        int[] ids = tokenizer.encode(text);
        int[] mask = tokenizer.attentionMask(ids);
        int[] segments = tokenizer.tokenTypeIds();

        long[][] inputIds = new long[1][ids.length];
        long[][] attentionMask = new long[1][mask.length];
        long[][] tokenTypes = new long[1][segments.length];

        for (int i = 0; i < ids.length; i++) {
            inputIds[0][i] = ids[i];
            attentionMask[0][i] = mask[i];
            tokenTypes[0][i] = segments[i];
        }

        try (
            OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
            OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
            OnnxTensor typesTensor = OnnxTensor.createTensor(env, tokenTypes)
        ) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", idsTensor);
            inputs.put("attention_mask", maskTensor);
            inputs.put("token_type_ids", typesTensor);

            try (OrtSession.Result output = session.run(inputs)) {

                float[][] logits = (float[][]) output.get(0).getValue();
                double[] probabilities = softmax(logits[0]);

                int predicted = 0;
                for (int i = 1; i < probabilities.length; i++) {
                    if (probabilities[i] > probabilities[predicted]) {
                        predicted = i;
                    }
                }

                return new Prediction(predicted, probabilities);
            }
        }
    }

    static double[] softmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;

        for (float value : logits) {
            max = Math.max(max, value);
        }

        double sum = 0;
        double[] result = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            result[i] = Math.exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    static int getLabelIndex(String label) {
        for (int i = 0; i < LABELS.length; i++) {
            if (LABELS[i].equalsIgnoreCase(label.trim())) {
                return i;
            }
        }
        return -1;
    }

    public static void main(String[] args) throws Exception {

        Files.createDirectories(Paths.get(RESULTS_PATH));

        System.out.println("========================================");
        System.out.println("MENTAL HEALTH BERT - JAVA EVALUATION");
        System.out.println("========================================");

        tokenizer = new Tokenizer(VOCAB_PATH);
        env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(MODEL_PATH, options);

        int[][] confusion = new int[4][4];
        int total = 0;
        int correct = 0;
        int skipped = 0;

        try (
            CSVReader reader = new CSVReader(new FileReader(TEST_PATH));
            CSVWriter predictionsWriter = new CSVWriter(
                new FileWriter(RESULTS_PATH + "/model_predictions.csv")
            );
            CSVWriter errorsWriter = new CSVWriter(
                new FileWriter(RESULTS_PATH + "/misclassified_examples.csv")
            )
        ) {

            predictionsWriter.writeNext(new String[]{
                "text", "actual_label", "predicted_label",
                "confidence", "Anxiety", "Depression", "Normal", "Suicidal"
            });

            errorsWriter.writeNext(new String[]{
                "text", "actual_label", "predicted_label", "confidence"
            });

            reader.readNext(); // Skip CSV header

            String[] row;

            while ((row = reader.readNext()) != null) {

                if (row.length < 2) {
                    skipped++;
                    continue;
                }

                String text = row[0];
                String actualLabel = row[1];

                int actual = getLabelIndex(actualLabel);

                if (actual == -1 || text == null || text.trim().isEmpty()) {
                    skipped++;
                    continue;
                }

                Prediction prediction = predict(text);
                int predicted = prediction.predicted;

                confusion[actual][predicted]++;

                total++;
                if (actual == predicted) {
                    correct++;
                }

                double confidence = prediction.probabilities[predicted];

                predictionsWriter.writeNext(new String[]{
                    text,
                    LABELS[actual],
                    LABELS[predicted],
                    String.format(Locale.US, "%.6f", confidence),
                    String.format(Locale.US, "%.6f", prediction.probabilities[0]),
                    String.format(Locale.US, "%.6f", prediction.probabilities[1]),
                    String.format(Locale.US, "%.6f", prediction.probabilities[2]),
                    String.format(Locale.US, "%.6f", prediction.probabilities[3])
                });

                if (actual != predicted) {
                    errorsWriter.writeNext(new String[]{
                        text,
                        LABELS[actual],
                        LABELS[predicted],
                        String.format(Locale.US, "%.6f", confidence)
                    });
                }

                if (total % 100 == 0) {
                    System.out.println("Processed " + total + " samples...");
                }
            }
        }

        double accuracy = total == 0 ? 0 : (double) correct / total;

        double macroPrecision = 0;
        double macroRecall = 0;
        double macroF1 = 0;

        StringBuilder report = new StringBuilder();

        report.append("MENTAL HEALTH BERT - CLASSIFICATION REPORT\n");
        report.append("============================================\n\n");
        report.append(String.format(Locale.US, "Total evaluated: %d\n", total));
        report.append(String.format(Locale.US, "Skipped rows: %d\n", skipped));
        report.append(String.format(Locale.US, "Accuracy: %.4f (%.2f%%)\n\n",
                accuracy, accuracy * 100));

        report.append(String.format(
            "%-15s %12s %12s %12s %12s\n",
            "Class", "Precision", "Recall", "F1-score", "Support"
        ));

        for (int c = 0; c < 4; c++) {

            int tp = confusion[c][c];
            int support = 0;
            int predictedCount = 0;

            for (int j = 0; j < 4; j++) {
                support += confusion[c][j];
                predictedCount += confusion[j][c];
            }

            double precision = predictedCount == 0 ? 0 :
                    (double) tp / predictedCount;

            double recall = support == 0 ? 0 :
                    (double) tp / support;

            double f1 = (precision + recall) == 0 ? 0 :
                    2 * precision * recall / (precision + recall);

            macroPrecision += precision;
            macroRecall += recall;
            macroF1 += f1;

            report.append(String.format(
                Locale.US, "%-15s %12.4f %12.4f %12.4f %12d\n",
                LABELS[c], precision, recall, f1, support
            ));
        }

        macroPrecision /= 4;
        macroRecall /= 4;
        macroF1 /= 4;

        report.append(String.format(
            Locale.US,
            "\nMacro Precision: %.4f\nMacro Recall: %.4f\nMacro F1-score: %.4f\n",
            macroPrecision, macroRecall, macroF1
        ));

        report.append("\nCONFUSION MATRIX\n");
        report.append("Actual rows / Predicted columns\n\n");

        report.append(String.format("%-15s", "Actual"));

        for (String label : LABELS) {
            report.append(String.format("%12s", label));
        }

        report.append("\n");

        for (int i = 0; i < 4; i++) {
            report.append(String.format("%-15s", LABELS[i]));

            for (int j = 0; j < 4; j++) {
                report.append(String.format("%12d", confusion[i][j]));
            }

            report.append("\n");
        }

        Files.write(
            Paths.get(RESULTS_PATH + "/classification_report.txt"),
            report.toString().getBytes()
        );

        try (CSVWriter matrixWriter = new CSVWriter(
                new FileWriter(RESULTS_PATH + "/confusion_matrix.csv"))) {

            String[] header = new String[5];
            header[0] = "Actual / Predicted";

            for (int i = 0; i < 4; i++) {
                header[i + 1] = LABELS[i];
            }

            matrixWriter.writeNext(header);

            for (int i = 0; i < 4; i++) {
                String[] matrixRow = new String[5];
                matrixRow[0] = LABELS[i];

                for (int j = 0; j < 4; j++) {
                    matrixRow[j + 1] = String.valueOf(confusion[i][j]);
                }

                matrixWriter.writeNext(matrixRow);
            }
        }

        System.out.println("\n" + report);
        System.out.println("========================================");
        System.out.println("Evaluation completed!");
        System.out.println("Results saved in: " + RESULTS_PATH);
        System.out.println("========================================");

        session.close();
        options.close();
    }
}
