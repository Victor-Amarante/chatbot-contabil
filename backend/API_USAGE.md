# API Usage Guide

## Endpoints Disponíveis

### 1. Upload de PDF
**POST** `/upload-pdf`

Faz upload de um arquivo PDF e inicializa o sistema RAG.

**Parâmetros:**
- `file`: Arquivo PDF (multipart/form-data)

**Resposta de sucesso:**
```json
{
    "message": "PDF carregado com sucesso! Agora você pode fazer perguntas sobre o documento.",
    "filename": "documento.pdf"
}
```

**Exemplo de uso com curl:**
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@seu_arquivo.pdf"
```

### 2. Fazer Pergunta
**POST** `/question`

Faz uma pergunta sobre o PDF carregado.

**Parâmetros:**
```json
{
    "question": "Sua pergunta aqui"
}
```

**Resposta:**
```json
{
    "answer": "Resposta baseada no conteúdo do PDF"
}
```

### 3. Verificar Status
**GET** `/status`

Verifica o status atual do sistema.

**Resposta:**
```json
{
    "pdf_loaded": true,
    "current_pdf": "/path/to/uploaded/file.pdf",
    "rag_initialized": true
}
```

## Fluxo de Uso

1. **Upload do PDF**: Primeiro, faça upload de um PDF usando o endpoint `/upload-pdf`
2. **Verificar Status**: Use `/status` para confirmar que o PDF foi carregado
3. **Fazer Perguntas**: Use `/question` para fazer perguntas sobre o documento

## Exemplo Completo

```bash
# 1. Upload do PDF
curl -X POST "http://localhost:8000/upload-pdf" \
     -F "file=@documento.pdf"

# 2. Verificar status
curl -X GET "http://localhost:8000/status"

# 3. Fazer uma pergunta
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "Qual é o tema principal do documento?"}'
```

## Notas Importantes

- Apenas arquivos PDF são aceitos no upload
- O sistema suporta apenas um PDF por vez
- Se um novo PDF for enviado, ele substituirá o anterior
- O sistema RAG é inicializado automaticamente após o upload bem-sucedido
- Todas as perguntas são salvas no banco de dados 