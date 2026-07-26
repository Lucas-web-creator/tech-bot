import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Modality } from '@google/genai';
import dotenv from 'dotenv';
import { CustomGPT, ChatGPTModel } from './src/types';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '25mb' }));

// Featured Custom GPTs Store
const defaultCustomGpts: CustomGPT[] = [
  {
    id: 'gpt-dalle',
    name: 'DALL·E 3',
    description: 'Transforme suas ideias em ilustrações, fotos conceituais e arte digital em alta resolução.',
    icon: '🎨',
    instructions: 'Você é um especialista em geração de imagens DALL-E 3. Crie descrições visuais ricas e gere imagens impressionantes sempre que o usuário solicitar uma criação visual.',
    author: 'OpenAI',
    category: 'Estilo de Vida',
    capabilities: { webSearch: false, imageGen: true, codeInterpreter: false },
    starterPrompts: [
      'Crie a ilustração de uma cidade futurista ao pôr do sol em estilo Cyberpunk',
      'Gere a foto de um gato astronauta explorando Marte',
      'Desenhe um logotipo minimalista para uma cafeteria moderna'
    ]
  },
  {
    id: 'gpt-code',
    name: 'Code Copilot Pro',
    description: 'Especialista em React, Node.js, Python, arquitetura de software, refatoração e otimização.',
    icon: '💻',
    instructions: 'Você é um especialista sênior em engenharia de software e programação. Escreva código limpo, moderno, tipado e com explicações didáticas. Sempre que criar um arquivo ou projeto extenso, utilize o Canvas.',
    author: 'OpenAI',
    category: 'Programação',
    capabilities: { webSearch: true, imageGen: false, codeInterpreter: true },
    starterPrompts: [
      'Como criar um hook customizado no React para gerenciar WebSocket?',
      'Refatore este algoritmo em Python para complexidade O(n)',
      'Crie um servidor Express com autenticação JWT e validação Zod'
    ]
  },
  {
    id: 'gpt-writer',
    name: 'Escritor & Copywriter',
    description: 'Aprimore e-mails, artigos para blog, redações acadêmicas, postagens e copys de vendas.',
    icon: '✍️',
    instructions: 'Você é um redator e copywriter premiado. Crie textos envolventes, claros, com excelente cadência e forte poder de persuasão.',
    author: 'OpenAI',
    category: 'Escrita',
    capabilities: { webSearch: true, imageGen: false, codeInterpreter: false },
    starterPrompts: [
      'Escreva um e-mail de lançamento para um novo aplicativo móvel',
      'Reescreva este parágrafo em um tom mais profissional e persuasivo',
      'Crie 5 títulos virais para um artigo sobre inteligência artificial'
    ]
  },
  {
    id: 'gpt-data',
    name: 'Analista de Dados & Math',
    description: 'Resolva problemas complexos de matemática, equações, raciocínio estatístico e análise.',
    icon: '📊',
    instructions: 'Você é um cientista de dados e matemático rigoroso. Mostre o passo a passo lógico para cada solução com clareza.',
    author: 'OpenAI',
    category: 'Produtividade',
    capabilities: { webSearch: true, imageGen: false, codeInterpreter: true },
    starterPrompts: [
      'Resolva a equação diferencial dy/dx + 2y = x',
      'Analise como calcular a taxa de retenção de clientes (Churn Rate)',
      'Explique o Teorema de Bayes com um exemplo prático'
    ]
  },
  {
    id: 'gpt-tutor',
    name: 'Tutor de Idiomas',
    description: 'Pratique conversação em Inglês, Espanhol, Francês ou Alemão com correção instantânea.',
    icon: '🌐',
    instructions: 'Você é um tutor bilíngue atencioso. Pratique diálogos naturais, corrija erros gramaticais delicadamente e dê sugestões de vocabulário mais nativo.',
    author: 'OpenAI',
    category: 'Educação',
    capabilities: { webSearch: false, imageGen: false, codeInterpreter: false },
    starterPrompts: [
      'Let\'s practice a job interview in English',
      'Me ensine 10 expressões idiomáticas comuns em espanhol',
      'Corrija as frases do meu e-mail em francês'
    ]
  }
];

let customGptsStore = [...defaultCustomGpts];

// Initialize Gemini API client
const getAiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("A chave GEMINI_API_KEY não foi configurada no ambiente.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build'
      }
    }
  });
};

// ChatGPT System Instruction Builder
const getChatGPTSystemInstruction = (
  model: ChatGPTModel,
  customInstructions?: { aboutUser?: string; responseStyle?: string },
  customGpt?: CustomGPT,
  enableReasoning?: boolean
) => {
  let instruction = `
Você é o **ChatGPT**, um modelo de inteligência artificial de grande porte treinado pela **OpenAI**.
Seu objetivo é ser extremamente prestativo, claro, natural, amigável e preciso em todas as interações.

### DIRETRIZES DE ESTILO DO CHATGPT:
1. Responda em Português do Brasil (a menos que o usuário escreva em outro idioma).
2. Utilize formatação Markdown rica e elegante (títulos, listas com marcadores, código formatado com realce de sintaxe).
3. Seja conciso e direto quando solicitado, mas completo e aprofundado em explicações detalhadas.
4. Mantenha um tom profissional, amigável e altamente capacitado.

### RECURSO CANVAS DO CHATGPT (CANVAS EDITOR):
Quando o usuário pedir para criar ou editar um documento extenso, código-fonte completo de uma aplicação, componente React, arquivo HTML/CSS/JS autocontido, artigo longo ou guia completo:
Você DEVE envolver essa parte do documento dentro da tag especial <chatgptCanvas>:

\`\`\`xml
<chatgptCanvas identifier="documento-id" type="code" title="Título do Documento" language="typescript">
// Conteúdo do arquivo ou código aqui...
</chatgptCanvas>
\`\`\`

Valores permitidos para o atributo \`type\`:
- \`code\` (para código em Python, JS, TS, React, HTML, CSS, C++, etc.)
- \`text\` ou \`markdown\` (para textos, artigos, ensaios, relatórios)
- \`html\` (para aplicações ou protótipos web visualizáveis)
- \`react\` (para componentes interativos em React)

Explicações curtas ou respostas normais devem ficar FORA do bloco <chatgptCanvas>. O conteúdo no Canvas abrirá em um editor lateral interativo no estilo do ChatGPT Canvas!
`;

  if (model === 'o1' || model === 'o3-mini' || enableReasoning) {
    instruction += `
### MODO RACIOCÍNIO / REASONING (O1 STYLE):
Você está operando com a capacidade de raciocínio lógico avançado ativada.
Antes de dar a resposta final ao usuário, estruture todo o seu raciocínio detalhado dentro de um bloco <reasoning time="5">...</reasoning> no início da resposta.
Exemplo:
<reasoning time="4">
1. Identificando a questão central do usuário...
2. Analisando as restrições e casos de borda...
3. Calculando os passos lógicos e verificando os resultados...
</reasoning>
Em seguida, apresente a solução limpa e definitiva.
`;
  }

  if (customGpt) {
    instruction += `
### MODO GPT CUSTOMIZADO: "${customGpt.name}"
Descrição: ${customGpt.description}
Instruções Especiais do GPT:
${customGpt.instructions}
`;
  }

  if (customInstructions) {
    if (customInstructions.aboutUser) {
      instruction += `\n### SOBRE O USUÁRIO (Personalização):\n${customInstructions.aboutUser}\n`;
    }
    if (customInstructions.responseStyle) {
      instruction += `\n### COMO O CHATGPT DEVE RESPONDER (Estilo de Resposta Personalizado):\n${customInstructions.responseStyle}\n`;
    }
  }

  return instruction;
};

// API ROUTES

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'ChatGPT Engine Server' });
});

// Custom GPTs API
app.get('/api/gpts', (req, res) => {
  res.json({ gpts: customGptsStore });
});

app.post('/api/gpts', (req, res) => {
  const { name, description, instructions, category, starterPrompts, icon } = req.body;
  if (!name || !instructions) {
    return res.status(400).json({ error: 'Nome e instruções do GPT são obrigatórios.' });
  }

  const newGpt: CustomGPT = {
    id: `gpt-custom-${Date.now()}`,
    name,
    description: description || 'GPT Personalizado',
    icon: icon || '🤖',
    instructions,
    author: 'Usuário',
    category: category || 'Geral',
    capabilities: { webSearch: true, imageGen: true, codeInterpreter: true },
    starterPrompts: starterPrompts || ['Como você pode me ajudar hoje?']
  };

  customGptsStore.unshift(newGpt);
  res.status(201).json({ success: true, gpt: newGpt });
});

// Image Generation Endpoint (DALL-E 3 Style)
app.post('/api/generate-image', async (req, res) => {
  try {
    const { prompt, aspectRatio = '1:1' } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'O prompt para geração de imagem é obrigatório.' });
    }

    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite-image',
      contents: {
        parts: [{ text: `Generate a high quality, detailed, artistic image based on this description: ${prompt}` }]
      },
      config: {
        imageConfig: {
          aspectRatio: aspectRatio as any || '1:1'
        }
      }
    });

    let imageUrl = null;
    if (response.candidates?.[0]?.content?.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData) {
          imageUrl = `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
          break;
        }
      }
    }

    if (!imageUrl) {
      return res.status(500).json({ error: 'Não foi possível gerar a imagem no momento.' });
    }

    res.json({ success: true, imageUrl, prompt });
  } catch (err: any) {
    console.error('Erro na geração de imagem DALL-E:', err);
    res.status(500).json({ error: 'Erro ao criar imagem.', details: err?.message });
  }
});

// Main ChatGPT Chat Endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const {
      messages,
      model = 'gpt-4o',
      enableWebSearch = false,
      enableReasoning = false,
      customInstructions,
      gptId,
      attachments
    } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Formato de mensagens inválido.' });
    }

    const ai = getAiClient();

    // Determine target Gemini Model
    // o1 / o3-mini reasoning -> gemini-3.1-pro-preview
    // gpt-4o / canvas -> gemini-3.6-flash
    // gpt-4o-mini -> gemini-3.6-flash with low temperature
    let selectedModel = 'gemini-3.6-flash';
    let temp = 0.7;

    if (model === 'o1' || model === 'o3-mini') {
      selectedModel = 'gemini-3.1-pro-preview';
      temp = 0.4;
    } else if (model === 'gpt-4o-mini') {
      selectedModel = 'gemini-3.6-flash';
      temp = 0.5;
    }

    const customGpt = gptId ? customGptsStore.find(g => g.id === gptId) : undefined;
    const systemInstruction = getChatGPTSystemInstruction(
      model as ChatGPTModel,
      customInstructions,
      customGpt,
      enableReasoning || model === 'o1' || model === 'o3-mini'
    );

    // Format chat contents
    const formattedContents = messages.map((m: { sender: string; text: string; attachments?: any[] }) => {
      let contentText = m.text;

      if (m.attachments && m.attachments.length > 0) {
        const attsText = m.attachments.map((a: any) =>
          `[Anexo Anexado: ${a.name} (${a.type})\nConteúdo: ${a.content || a.name}]`
        ).join('\n');
        contentText = `${attsText}\n\n${contentText}`;
      }

      return {
        role: m.sender === 'user' ? 'user' : 'model',
        parts: [{ text: contentText }]
      };
    });

    // Check if user is asking to generate an image explicitly
    const lastUserMsg = messages[messages.length - 1]?.text?.toLowerCase() || '';
    const isImageRequest = lastUserMsg.includes('gere uma imagem') ||
                           lastUserMsg.includes('crie uma imagem') ||
                           lastUserMsg.includes('desenhe') ||
                           lastUserMsg.includes('dall-e') ||
                           lastUserMsg.includes('generate an image') ||
                           lastUserMsg.includes('faça uma imagem');

    if (isImageRequest && (customGpt?.capabilities.imageGen !== false)) {
      try {
        const imgResponse = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite-image',
          contents: {
            parts: [{ text: `Create a beautiful image of: ${messages[messages.length - 1].text}` }]
          },
          config: {
            imageConfig: { aspectRatio: '1:1' }
          }
        });

        let generatedImgUrl = null;
        if (imgResponse.candidates?.[0]?.content?.parts) {
          for (const part of imgResponse.candidates[0].content.parts) {
            if (part.inlineData) {
              generatedImgUrl = `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
              break;
            }
          }
        }

        if (generatedImgUrl) {
          return res.json({
            text: `Aqui está a imagem que criei para você com o DALL·E 3:`,
            imageUrl: generatedImgUrl,
            modelUsed: model
          });
        }
      } catch (imgErr) {
        console.error('Falha ao tentar gerar imagem via chat:', imgErr);
        // Fallback to text conversation
      }
    }

    // Config tools for Web Search grounding if enabled
    const tools: any[] = [];
    if (enableWebSearch || customGpt?.capabilities.webSearch) {
      tools.push({ googleSearch: {} });
    }

    const response = await ai.models.generateContent({
      model: selectedModel,
      contents: formattedContents,
      config: {
        systemInstruction,
        temperature: temp,
        ...(tools.length > 0 ? { tools } : {})
      }
    });

    const fullText = response.text || "O ChatGPT não retornou resposta no momento.";

    // Parse Search Grounding sources if web search was used
    let sources = undefined;
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (groundingChunks && Array.isArray(groundingChunks)) {
      sources = groundingChunks
        .filter((chunk: any) => chunk.web)
        .map((chunk: any) => ({
          title: chunk.web.title || 'Fonte Web',
          url: chunk.web.uri,
          snippet: chunk.web.snippet
        }))
        .slice(0, 5);
    }

    // Parse Reasoning block if present (<reasoning time="5">...</reasoning>)
    let reasoning = undefined;
    let reasoningTimeSeconds = undefined;
    let cleanText = fullText;

    const reasoningMatch = fullText.match(/<reasoning(?:\s+time="(\d+)")?>([\s\S]*?)<\/reasoning>/);
    if (reasoningMatch) {
      reasoningTimeSeconds = reasoningMatch[1] ? parseInt(reasoningMatch[1], 10) : 3;
      reasoning = reasoningMatch[2].trim();
      cleanText = fullText.replace(/<reasoning[\s\S]*?<\/reasoning>/, '').trim();
    }

    // Parse Canvas block if present (<chatgptCanvas identifier="..." type="..." title="...">...</chatgptCanvas>)
    let canvas = undefined;
    const canvasMatch = cleanText.match(/<chatgptCanvas\s+identifier="([^"]+)"\s+type="([^"]+)"\s+title="([^"]+)"(?:\s+language="([^"]+)")?>([\s\S]*?)<\/chatgptCanvas>/);

    if (canvasMatch) {
      const [, identifier, type, title, language, content] = canvasMatch;

      let canvasType: 'code' | 'text' | 'html' | 'markdown' | 'react' = 'code';
      if (type.includes('react')) canvasType = 'react';
      else if (type.includes('html')) canvasType = 'html';
      else if (type.includes('text') || type.includes('markdown')) canvasType = 'text';

      canvas = {
        id: identifier || `canvas-${Date.now()}`,
        title: title || 'Documento do Canvas',
        type: canvasType,
        language: language || 'typescript',
        content: content.trim(),
        version: 1
      };

      cleanText = cleanText.replace(/<chatgptCanvas[\s\S]*?<\/chatgptCanvas>/, '').trim();
      if (!cleanText) {
        cleanText = `Criei o documento **"${title}"** no Canvas. Você pode visualizar e editá-lo lado a lado!`;
      } else {
        cleanText += `\n\n*(Documento **"${title}"** aberto no Canvas)*`;
      }
    }

    res.json({
      text: cleanText,
      reasoning,
      reasoningTimeSeconds,
      canvas,
      sources,
      modelUsed: model
    });

  } catch (err: any) {
    console.error('Erro no endpoint do ChatGPT:', err);
    res.status(500).json({
      error: 'Erro na resposta do ChatGPT.',
      details: err?.message || 'Falha de comunicação no servidor'
    });
  }
});

// TTS Endpoint (Voice response)
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'alloy' } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Texto é necessário.' });
    }

    const ai = getAiClient();
    const voiceMap: Record<string, string> = {
      alloy: 'Kore',
      echo: 'Fenrir',
      fable: 'Puck',
      onyx: 'Charon',
      nova: 'Zephyr',
      shimmer: 'Kore'
    };

    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-tts-preview',
      contents: [{ parts: [{ text: `Leia em tom amigável e natural: ${text.slice(0, 500)}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: voiceMap[voice] || 'Kore' }
          }
        }
      }
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      return res.json({ audioBase64: base64Audio });
    } else {
      return res.status(500).json({ error: 'Áudio não pôde ser gerado.' });
    }
  } catch (err: any) {
    console.error('Erro no TTS:', err);
    res.status(500).json({ error: 'Erro de voz.', details: err?.message });
  }
});

// Vite server for development & static for production
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Servidor ChatGPT rodando na porta ${PORT}`);
  });
}

startServer();
