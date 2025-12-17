#include "tree_sitter/alloc.h"
#include "tree_sitter/parser.h"

enum TokenType {
    // Raw string (no escape, no interpolation) - single token
    RAW_STRING_LITERAL,
    // Raw bytes (no escape, no interpolation) - single token
    RAW_BYTES_LITERAL,
    // Block comment with nesting support: /* ... /* nested */ ... */
    BLOCK_COMMENT,
    // Block doc comment with nesting support: /** ... */
    BLOCK_DOC_COMMENT,
    // Multiline string with interpolation: #"...\{expr}..."#
    MULTILINE_STRING_START,    // #"
    MULTILINE_STRING_CONTENT,  // text between interpolations
    MULTILINE_STRING_END,      // "#
    // Multiline bytes with interpolation: b#"...\{expr}..."#
    MULTILINE_BYTES_START,     // b#"
    MULTILINE_BYTES_CONTENT,   // text between interpolations
    MULTILINE_BYTES_END,       // "#
    // Newline token for field separators (Go/Swift style)
    NEWLINE,

    ERROR_SENTINEL
};

// Scanner state for multiline strings with hash delimiters
typedef struct {
    uint8_t opening_hash_count;
    bool in_multiline_string;
    bool in_multiline_bytes;
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
    buffer[1] = (char)(scanner->in_multiline_string ? 1 : 0);
    buffer[2] = (char)(scanner->in_multiline_bytes ? 1 : 0);
    return 3;
}

void tree_sitter_tribute_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
    Scanner *scanner = (Scanner *)payload;
    scanner->opening_hash_count = 0;
    scanner->in_multiline_string = false;
    scanner->in_multiline_bytes = false;
    if (length >= 3) {
        scanner->opening_hash_count = (uint8_t)buffer[0];
        scanner->in_multiline_string = buffer[1] != 0;
        scanner->in_multiline_bytes = buffer[2] != 0;
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

// Scan multiline string/bytes content until interpolation or end delimiter
// Returns true if content was found (even empty content before \{ or end)
static bool scan_multiline_content(TSLexer *lexer, Scanner *scanner, bool is_bytes) {
    uint8_t hash_count = scanner->opening_hash_count;
    bool has_content = false;

    for (;;) {
        if (lexer->eof(lexer)) {
            // Unterminated - return what we have
            if (has_content) {
                lexer->result_symbol = is_bytes ? MULTILINE_BYTES_CONTENT : MULTILINE_STRING_CONTENT;
                lexer->mark_end(lexer);
                return true;
            }
            return false;
        }

        // Check for interpolation start: \{
        if (lexer->lookahead == '\\') {
            lexer->mark_end(lexer);
            advance(lexer);
            if (lexer->lookahead == '{') {
                // Don't consume \{ - let grammar handle it
                // Return content we've accumulated (may be empty)
                lexer->result_symbol = is_bytes ? MULTILINE_BYTES_CONTENT : MULTILINE_STRING_CONTENT;
                return true;
            }
            // Not interpolation, backslash is part of content
            has_content = true;
            continue;
        }

        // Check for end delimiter: "# (with matching hash count)
        if (lexer->lookahead == '"') {
            lexer->mark_end(lexer);
            advance(lexer);

            // Count closing hashes
            uint8_t closing_hash_count = 0;
            while (lexer->lookahead == '#' && closing_hash_count < hash_count) {
                advance(lexer);
                closing_hash_count++;
            }

            // If we matched all hashes, this is the end
            if (closing_hash_count == hash_count) {
                // Return content first (don't consume the end delimiter)
                // The end will be returned on next scan
                lexer->result_symbol = is_bytes ? MULTILINE_BYTES_CONTENT : MULTILINE_STRING_CONTENT;
                return true;
            }
            // Not the end, quote and hashes are content
            has_content = true;
            continue;
        }

        // Regular content character
        advance(lexer);
        has_content = true;
        lexer->mark_end(lexer);
    }
}

// Scan multiline string/bytes end delimiter
static bool scan_multiline_end(TSLexer *lexer, Scanner *scanner, bool is_bytes) {
    uint8_t hash_count = scanner->opening_hash_count;

    if (lexer->lookahead != '"') {
        return false;
    }
    advance(lexer);

    // Count closing hashes
    uint8_t closing_hash_count = 0;
    while (lexer->lookahead == '#' && closing_hash_count < hash_count) {
        advance(lexer);
        closing_hash_count++;
    }

    if (closing_hash_count == hash_count) {
        lexer->result_symbol = is_bytes ? MULTILINE_BYTES_END : MULTILINE_STRING_END;
        lexer->mark_end(lexer);
        // Clear state
        scanner->opening_hash_count = 0;
        scanner->in_multiline_string = false;
        scanner->in_multiline_bytes = false;
        return true;
    }

    return false;
}

// Scan block comment with nesting support
static bool scan_block_comment(TSLexer *lexer, bool is_doc_comment) {
    int nesting_depth = 1;

    while (nesting_depth > 0) {
        if (lexer->eof(lexer)) {
            return false;
        }

        if (lexer->lookahead == '/') {
            advance(lexer);
            if (lexer->lookahead == '*') {
                advance(lexer);
                nesting_depth++;
            }
        } else if (lexer->lookahead == '*') {
            advance(lexer);
            if (lexer->lookahead == '/') {
                advance(lexer);
                nesting_depth--;
            }
        } else {
            advance(lexer);
        }
    }

    lexer->result_symbol = is_doc_comment ? BLOCK_DOC_COMMENT : BLOCK_COMMENT;
    lexer->mark_end(lexer);
    return true;
}

bool tree_sitter_tribute_external_scanner_scan(
    void *payload,
    TSLexer *lexer,
    const bool *valid_symbols
) {
    Scanner *scanner = (Scanner *)payload;

    // Error recovery mode - bail out
    if (valid_symbols[ERROR_SENTINEL]) {
        return false;
    }

    // If we're inside a multiline string, handle content/end
    if (scanner->in_multiline_string) {
        // Try to scan end first
        if (valid_symbols[MULTILINE_STRING_END] && lexer->lookahead == '"') {
            if (scan_multiline_end(lexer, scanner, false)) {
                return true;
            }
        }
        // Scan content
        if (valid_symbols[MULTILINE_STRING_CONTENT]) {
            return scan_multiline_content(lexer, scanner, false);
        }
        return false;
    }

    // If we're inside multiline bytes, handle content/end
    if (scanner->in_multiline_bytes) {
        // Try to scan end first
        if (valid_symbols[MULTILINE_BYTES_END] && lexer->lookahead == '"') {
            if (scan_multiline_end(lexer, scanner, true)) {
                return true;
            }
        }
        // Scan content
        if (valid_symbols[MULTILINE_BYTES_CONTENT]) {
            return scan_multiline_content(lexer, scanner, true);
        }
        return false;
    }

    // Handle whitespace
    // Skip spaces and tabs always
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t') {
        skip(lexer);
    }

    // If NEWLINE token is valid (e.g., in struct fields), emit it
    // BUT only if followed by something that looks like another field (identifier)
    if (valid_symbols[NEWLINE] && (lexer->lookahead == '\n' || lexer->lookahead == '\r')) {
        // Mark position before consuming newline
        lexer->mark_end(lexer);

        // Consume newline(s) and whitespace
        while (lexer->lookahead == '\n' || lexer->lookahead == '\r' ||
               lexer->lookahead == ' ' || lexer->lookahead == '\t') {
            advance(lexer);
        }

        // Check if next char looks like start of identifier or type name
        // (lowercase letter, uppercase letter, or underscore)
        // This prevents consuming newline when we're at the end of fields (before '}')
        if ((lexer->lookahead >= 'a' && lexer->lookahead <= 'z') ||
            (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z') ||
            lexer->lookahead == '_') {
            lexer->result_symbol = NEWLINE;
            lexer->mark_end(lexer);
            return true;
        }
        // Not followed by identifier - don't emit NEWLINE, let normal whitespace skipping handle it
        // But we've already consumed the whitespace, so just continue
    }

    // Otherwise skip newlines as normal whitespace
    while (lexer->lookahead == '\n' || lexer->lookahead == '\r') {
        skip(lexer);
    }
    // Skip any remaining spaces/tabs after newlines
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t') {
        skip(lexer);
    }

    // Raw strings/bytes: r"...", r#"..."#, rb"...", rb#"..."#
    if (lexer->lookahead == 'r') {
        lexer->mark_end(lexer);
        advance(lexer);

        if (lexer->lookahead == 'b' && valid_symbols[RAW_BYTES_LITERAL]) {
            advance(lexer);
            if (lexer->lookahead == '#' || lexer->lookahead == '"') {
                return scan_raw_literal(lexer, RAW_BYTES_LITERAL);
            }
        }

        if (valid_symbols[RAW_STRING_LITERAL]) {
            if (lexer->lookahead == '#' || lexer->lookahead == '"') {
                return scan_raw_literal(lexer, RAW_STRING_LITERAL);
            }
        }
    }

    // Multiline bytes start: b#"
    if (lexer->lookahead == 'b' && valid_symbols[MULTILINE_BYTES_START]) {
        lexer->mark_end(lexer);
        advance(lexer);

        if (lexer->lookahead == '#') {
            uint8_t hash_count = 0;
            while (lexer->lookahead == '#') {
                advance(lexer);
                hash_count++;
            }

            if (lexer->lookahead == '"') {
                advance(lexer);
                lexer->result_symbol = MULTILINE_BYTES_START;
                lexer->mark_end(lexer);
                scanner->opening_hash_count = hash_count;
                scanner->in_multiline_bytes = true;
                return true;
            }
        }
    }

    // Multiline string start: #"
    if (lexer->lookahead == '#' && valid_symbols[MULTILINE_STRING_START]) {
        lexer->mark_end(lexer);

        uint8_t hash_count = 0;
        while (lexer->lookahead == '#') {
            advance(lexer);
            hash_count++;
        }

        if (lexer->lookahead == '"') {
            advance(lexer);
            lexer->result_symbol = MULTILINE_STRING_START;
            lexer->mark_end(lexer);
            scanner->opening_hash_count = hash_count;
            scanner->in_multiline_string = true;
            return true;
        }
    }

    // Block comments: /* ... */ or /** ... */
    if (lexer->lookahead == '/' &&
        (valid_symbols[BLOCK_COMMENT] || valid_symbols[BLOCK_DOC_COMMENT])) {
        lexer->mark_end(lexer);
        advance(lexer);

        if (lexer->lookahead == '*') {
            advance(lexer);

            if (lexer->lookahead == '*' && valid_symbols[BLOCK_DOC_COMMENT]) {
                advance(lexer);
                if (lexer->lookahead == '/') {
                    advance(lexer);
                    lexer->result_symbol = BLOCK_COMMENT;
                    lexer->mark_end(lexer);
                    return true;
                }
                return scan_block_comment(lexer, true);
            }

            if (valid_symbols[BLOCK_COMMENT]) {
                return scan_block_comment(lexer, false);
            }
        }
    }

    return false;
}
