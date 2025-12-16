#include "tree_sitter/alloc.h"
#include "tree_sitter/parser.h"

enum TokenType {
    // Raw string (no escape, no interpolation) - single token
    RAW_STRING_LITERAL,
    // Raw bytes (no escape, no interpolation) - single token
    RAW_BYTES_LITERAL,

    // Future: multiline strings with interpolation need start/content/end
    // STRING_START,        // #" or s#"
    // STRING_CONTENT,      // text between interpolations
    // STRING_END,          // "#
    // BYTES_START,         // b#"
    // BYTES_CONTENT,
    // BYTES_END,

    ERROR_SENTINEL
};

// Scanner state for multiline strings with hash delimiters
typedef struct {
    uint8_t opening_hash_count;
} Scanner;

void *tree_sitter_tribute_external_scanner_create(void) {
    return ts_calloc(1, sizeof(Scanner));
}

void tree_sitter_tribute_external_scanner_destroy(void *payload) {
    ts_free(payload);
}

unsigned tree_sitter_tribute_external_scanner_serialize(void *payload, char *buffer) {
    Scanner *scanner = (Scanner *)payload;
    buffer[0] = (char)scanner->opening_hash_count;
    return 1;
}

void tree_sitter_tribute_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
    Scanner *scanner = (Scanner *)payload;
    scanner->opening_hash_count = 0;
    if (length == 1) {
        scanner->opening_hash_count = (uint8_t)buffer[0];
    }
}

static inline void advance(TSLexer *lexer) {
    lexer->advance(lexer, false);
}

static inline void skip(TSLexer *lexer) {
    lexer->advance(lexer, true);
}

// Scan raw literal with hash delimiters
// Used for both raw strings (r"...", r#"..."#) and raw bytes (rb"...", rb#"..."#)
static bool scan_raw_literal(TSLexer *lexer, enum TokenType token_type) {
    // Count opening hashes
    uint8_t opening_hash_count = 0;
    while (lexer->lookahead == '#') {
        advance(lexer);
        opening_hash_count++;
    }

    // Must have opening quote
    if (lexer->lookahead != '"') {
        return false;
    }
    advance(lexer);

    // Scan content until we find closing quote + matching hashes
    for (;;) {
        if (lexer->eof(lexer)) {
            return false;
        }

        if (lexer->lookahead == '"') {
            advance(lexer);

            // Count closing hashes
            uint8_t closing_hash_count = 0;
            while (lexer->lookahead == '#' && closing_hash_count < opening_hash_count) {
                advance(lexer);
                closing_hash_count++;
            }

            // If we matched all hashes, we're done
            if (closing_hash_count == opening_hash_count) {
                lexer->result_symbol = token_type;
                lexer->mark_end(lexer);
                return true;
            }
            // Otherwise, the quote and hashes are part of the content, continue
        } else {
            advance(lexer);
        }
    }
}

bool tree_sitter_tribute_external_scanner_scan(
    void *payload,
    TSLexer *lexer,
    const bool *valid_symbols
) {
    (void)payload;  // Unused for now
    // Error recovery mode - bail out
    if (valid_symbols[ERROR_SENTINEL]) {
        return false;
    }

    // Skip whitespace
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
           lexer->lookahead == '\n' || lexer->lookahead == '\r') {
        skip(lexer);
    }

    // Both raw_string and raw_bytes start with 'r'
    if (lexer->lookahead == 'r') {
        // Check if it's rb" (raw bytes) or r" (raw string)
        // We need to peek ahead without committing
        lexer->mark_end(lexer);

        advance(lexer);  // consume 'r'

        if (lexer->lookahead == 'b' && valid_symbols[RAW_BYTES_LITERAL]) {
            advance(lexer);  // consume 'b'
            if (lexer->lookahead == '#' || lexer->lookahead == '"') {
                return scan_raw_literal(lexer, RAW_BYTES_LITERAL);
            }
        }

        // Not raw bytes, try raw string
        if (valid_symbols[RAW_STRING_LITERAL]) {
            if (lexer->lookahead == '#' || lexer->lookahead == '"') {
                return scan_raw_literal(lexer, RAW_STRING_LITERAL);
            }
        }
    }

    return false;
}
