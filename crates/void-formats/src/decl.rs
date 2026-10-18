//! Tokenizer for the text declaration syntax used by `.decl`, `.entities`,
//! and material files (`docs/formats/decl.md`).
//!
//! The syntax is brace-structured with `//` and `/* */` comments, quoted
//! strings, and bare words that may contain path characters. Materials use
//! newline-terminated `key value` lines; entity/component decls use
//! `key = value;`. The tokenizer records line numbers so both styles can be
//! parsed on top of it, and reports positions for malformed input.

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Word(String),
    Str(String),
    Open,
    Close,
    Eq,
    Semi,
    Comma,
    LParen,
    RParen,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
}

fn invalid(line: u32, detail: &str) -> Error {
    Error::Invalid {
        field: "decl text",
        detail: format!("line {line}: {detail}"),
    }
}

pub fn tokenize(text: &str) -> Result<Vec<Token>, Error> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut line) = (0usize, 1u32);
    while i < b.len() {
        let c = b[i];
        match c {
            b'\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let start = line;
                i += 2;
                loop {
                    if i + 1 >= b.len() {
                        return Err(invalid(start, "unterminated block comment"));
                    }
                    if b[i] == b'*' && b[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    if b[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
            b'"' => {
                let start = line;
                let mut s = Vec::new();
                i += 1;
                loop {
                    match b.get(i) {
                        None => return Err(invalid(start, "unterminated string")),
                        Some(b'"') => {
                            i += 1;
                            break;
                        }
                        Some(b'\\') if i + 1 < b.len() => {
                            s.push(b[i + 1]);
                            i += 2;
                        }
                        Some(&ch) => {
                            if ch == b'\n' {
                                line += 1;
                            }
                            s.push(ch);
                            i += 1;
                        }
                    }
                }
                let s = String::from_utf8(s).map_err(|_| invalid(start, "string is not UTF-8"))?;
                out.push(Token {
                    tok: Tok::Str(s),
                    line: start,
                });
            }
            b'{' | b'}' | b'=' | b';' | b',' | b'(' | b')' => {
                let tok = match c {
                    b'{' => Tok::Open,
                    b'}' => Tok::Close,
                    b'=' => Tok::Eq,
                    b';' => Tok::Semi,
                    b',' => Tok::Comma,
                    b'(' => Tok::LParen,
                    _ => Tok::RParen,
                };
                out.push(Token { tok, line });
                i += 1;
            }
            _ => {
                let start = i;
                while i < b.len() {
                    let ch = b[i];
                    if ch.is_ascii_whitespace() || b"{}=;,()\"".contains(&ch) {
                        break;
                    }
                    if ch == b'/' && matches!(b.get(i + 1), Some(b'/') | Some(b'*')) {
                        break;
                    }
                    i += 1;
                }
                let w = std::str::from_utf8(&b[start..i])
                    .map_err(|_| invalid(line, "word is not UTF-8"))?;
                out.push(Token {
                    tok: Tok::Word(w.to_owned()),
                    line,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(t: &[Token]) -> Vec<String> {
        t.iter()
            .map(|t| match &t.tok {
                Tok::Word(w) => w.clone(),
                Tok::Str(s) => format!("\"{s}\""),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn material_style() {
        let t = tokenize("{\n  diffusemap  models/a_d.tga // c\n  v { 0.3, 1 }\n}\n").unwrap();
        assert_eq!(
            words(&t),
            [
                "Open",
                "diffusemap",
                "models/a_d.tga",
                "v",
                "Open",
                "0.3",
                "Comma",
                "1",
                "Close",
                "Close"
            ]
        );
        assert_eq!(t[1].line, 2);
        assert_eq!(t[3].line, 3);
    }

    #[test]
    fn entity_style_and_comments() {
        let t = tokenize("a = \"x y\"; /* multi\nline */ b=1;").unwrap();
        assert_eq!(
            words(&t),
            ["a", "Eq", "\"x y\"", "Semi", "b", "Eq", "1", "Semi"]
        );
        assert_eq!(t[4].line, 2);
    }

    #[test]
    fn image_program_words() {
        let t = tokenize("map ipr_constantcolor( 0.5, 0.5, 0, 0)").unwrap();
        assert_eq!(words(&t)[1..3], ["ipr_constantcolor", "LParen"]);
    }

    #[test]
    fn rejects_unterminated() {
        assert!(tokenize("a \"b").is_err());
        assert!(tokenize("/* x").is_err());
    }
}
