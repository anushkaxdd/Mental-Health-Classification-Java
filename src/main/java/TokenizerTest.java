
import java.io.*;

public class TokenizerTest {

    public static void main(String[] args) throws Exception {

        Tokenizer tokenizer = new Tokenizer(
            "/content/MentalHealthBERT/models/vocab.txt"
        );

        String text = "I've been feeling really overwhelmed and anxious lately.";

        int[] ids = tokenizer.encode(text);
        int[] mask = tokenizer.attentionMask(ids);

        try (PrintWriter writer = new PrintWriter(
                "/content/MentalHealthBERT/results/java_token_ids.txt")) {

            for (int id : ids) {
                writer.print(id + " ");
            }
        }

        System.out.println("Text: " + text);
        System.out.println("Sequence length: " + ids.length);
        System.out.println("Java token IDs saved successfully.");
    }
}
