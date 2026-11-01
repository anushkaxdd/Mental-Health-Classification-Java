
import java.io.*;
import java.text.Normalizer;
import java.util.*;

public class Tokenizer {

    private Map<String, Integer> vocab = new HashMap<>();

    private static final int MAX_LENGTH = 128;
    private static final int PAD_ID = 0;
    private static final int UNK_ID = 100;
    private static final int CLS_ID = 101;
    private static final int SEP_ID = 102;

    public Tokenizer(String vocabPath) throws IOException {

        BufferedReader reader = new BufferedReader(
            new InputStreamReader(
                new FileInputStream(vocabPath),
                "UTF-8"
            )
        );

        String line;
        int index = 0;

        while ((line = reader.readLine()) != null) {
            vocab.put(line, index);
            index++;
        }

        reader.close();

        System.out.println("Vocabulary loaded: " + vocab.size());
    }

    private boolean isPunctuation(char c) {
        int type = Character.getType(c);

        return type == Character.CONNECTOR_PUNCTUATION ||
               type == Character.DASH_PUNCTUATION ||
               type == Character.START_PUNCTUATION ||
               type == Character.END_PUNCTUATION ||
               type == Character.OTHER_PUNCTUATION ||
               type == Character.INITIAL_QUOTE_PUNCTUATION ||
               type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    private boolean isCJK(int cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x3040 && cp <= 0x30FF) ||
               (cp >= 0xAC00 && cp <= 0xD7AF);
    }

    private String normalize(String text) {

        StringBuilder result = new StringBuilder();

        for (int i = 0; i < text.length();) {

            int cp = text.codePointAt(i);
            i += Character.charCount(cp);

            if (cp == 0 || cp == 0xFFFD ||
                (Character.isISOControl(cp) &&
                 !Character.isWhitespace(cp))) {
                continue;
            }

            if (Character.isWhitespace(cp) ||
                Character.isSpaceChar(cp)) {
                result.append(' ');
                continue;
            }

            if (isCJK(cp)) {
                result.append(' ').appendCodePoint(cp).append(' ');
            } else {
                result.appendCodePoint(cp);
            }
        }

        String lower = result.toString().toLowerCase(Locale.ROOT);

        String decomposed = Normalizer.normalize(
            lower, Normalizer.Form.NFD
        );

        StringBuilder clean = new StringBuilder();

        for (int i = 0; i < decomposed.length();) {

            int cp = decomposed.codePointAt(i);
            i += Character.charCount(cp);

            if (Character.getType(cp) != Character.NON_SPACING_MARK) {
                clean.appendCodePoint(cp);
            }
        }

        return clean.toString();
    }

    private List<String> basicTokenize(String text) {

        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        String normalized = normalize(text);

        for (int i = 0; i < normalized.length(); i++) {

            char c = normalized.charAt(i);

            if (Character.isWhitespace(c)) {

                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }

            } else if (isPunctuation(c) || isCJK(c)) {

                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }

                tokens.add(String.valueOf(c));

            } else {
                current.append(c);
            }
        }

        if (current.length() > 0) {
            tokens.add(current.toString());
        }

        return tokens;
    }

    private List<String> wordPiece(String token) {

        List<String> pieces = new ArrayList<>();

        if (token.length() > 100) {
            pieces.add("[UNK]");
            return pieces;
        }

        int start = 0;

        while (start < token.length()) {

            int end = token.length();
            String found = null;

            while (start < end) {

                String piece = token.substring(start, end);

                if (start > 0) {
                    piece = "##" + piece;
                }

                if (vocab.containsKey(piece)) {
                    found = piece;
                    break;
                }

                end--;
            }

            if (found == null) {
                pieces.clear();
                pieces.add("[UNK]");
                return pieces;
            }

            pieces.add(found);
            start = end;
        }

        return pieces;
    }

    public int[] encode(String text) {

        List<String> tokens = basicTokenize(text);
        List<Integer> ids = new ArrayList<>();

        ids.add(CLS_ID);

        for (String token : tokens) {

            for (String piece : wordPiece(token)) {

                if (ids.size() >= MAX_LENGTH - 1) {
                    break;
                }

                ids.add(vocab.getOrDefault(piece, UNK_ID));
            }

            if (ids.size() >= MAX_LENGTH - 1) {
                break;
            }
        }

        ids.add(SEP_ID);

        while (ids.size() < MAX_LENGTH) {
            ids.add(PAD_ID);
        }

        int[] result = new int[MAX_LENGTH];

        for (int i = 0; i < MAX_LENGTH; i++) {
            result[i] = ids.get(i);
        }

        return result;
    }

    public int[] attentionMask(int[] inputIds) {

        int[] mask = new int[MAX_LENGTH];

        for (int i = 0; i < MAX_LENGTH; i++) {
            mask[i] = inputIds[i] == PAD_ID ? 0 : 1;
        }

        return mask;
    }

    public int[] tokenTypeIds() {
        return new int[MAX_LENGTH];
    }
}
