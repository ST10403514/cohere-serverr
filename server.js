import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { CohereClient } from 'cohere-ai';
import fs from 'fs/promises';
import path from 'path';

const app = express();
app.use(cors());
app.use(express.json());

const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY
});
// DO NOT CHANGE
const EMBEDDINGS_FILE = './documents/embeddings.json';

function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magA * magB);
}

function getTopKDocuments(queryEmbedding, documents, k = 5) {
  const similarities = documents.map(doc => ({
    doc,
    score: cosineSimilarity(queryEmbedding, doc.embedding)
  }));

  similarities.sort((a, b) => b.score - a.score);
  return similarities.slice(0, k).map(item => item.doc);
}
//upload YOUR OWN DOICUMENTS HERE
async function loadDocuments() {
  const files = [
    './documents/tour_details.json',
    './documents/tours.json',
    './documents/rest_countries.json',
    './documents/unesco_sites.json',
    './documents/merged_countries.json'
  ];

  const documents = [];
//change to fit your documents
  
  for (const file of files) {
    try {
      const content = await fs.readFile(file, 'utf-8');
      const json = JSON.parse(content);

      if (file.includes('tour_details')) {
        json.forEach(tour => {
          const detailsSnippet = tour.details?.map(d => d.body).join(' ') || '';
          documents.push({
            id: `tour_details_${tour.name}`,
            data: {
              title: tour.name,
              snippet: (tour.description || '') + ' ' + detailsSnippet
            }
          });
        });
      } else if (file.includes('tours')) {
        json.forEach(tour => {
          documents.push({
            id: `tours_${tour.name}`,
            data: {
              title: tour.name,
              snippet: tour.product_line || ''
            }
          });
        });
      } else if (file.includes('rest_countries')) {
        json.forEach(country => {
          const languageList = Object.values(country.languages || {}).join(', ');
          documents.push({
            id: `rest_countries_${country.name?.common}`,
            data: {
              title: country.name?.common || '',
              snippet: `Official Name: ${country.name?.official}. Capital: ${country.capital?.[0] || ''}. Region: ${country.region}. Subregion: ${country.subregion}. Population: ${country.population}. Languages: ${languageList}. Area: ${country.area} sq km.`
            }
          });
        });
      } else if (file.includes('unesco_sites')) {
        json.query?.row?.forEach(site => {
          documents.push({
            id: `unesco_sites_${site.site}`,
            data: {
              title: site.site || '',
              snippet: site.short_description?.replace(/<[^>]+>/g, '') || ''
            }
          });
        });
      } else if (file.includes('merged_countries')) {
        json.forEach(country => {
          documents.push({
            id: `merged_countries_${country.name}`,
            data: {
              title: country.name || '',
              snippet: `Capital: ${country.capital}, Region: ${country.region}, Population: ${country.population}, Language: ${country.language}, Currency: ${country.currency}`
            }
          });
        });
      }

    } catch (err) {
      console.error(`Error reading ${file}:`, err);
    }
  }

  console.log(`Loaded ${documents.length} documents`);
  return documents;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function embedDocumentsInBatches(documents, batchSize = 96) {
  const allEmbeddings = [];

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    console.log(`Embedding batch ${i / batchSize + 1} of ${Math.ceil(documents.length / batchSize)}...`);

    const response = await cohere.embed({
      texts: batch.map(doc => `${doc.data.title}. ${doc.data.snippet}`),
      model: 'embed-multilingual-v3.0',
      input_type: 'search_document'
    });

    allEmbeddings.push(...response.embeddings);
    await sleep(10000);
  }

  return allEmbeddings;
}

async function saveEmbeddingsToFile(documents) {
  const embeddingsData = documents.map(doc => ({
    id: doc.id,
    title: doc.data.title,
    snippet: doc.data.snippet,
    embedding: doc.embedding
  }));

  await fs.writeFile(EMBEDDINGS_FILE, JSON.stringify(embeddingsData, null, 2), 'utf-8');
  console.log(`Embeddings saved to ${EMBEDDINGS_FILE}`);
}

async function loadEmbeddingsFromFile() {
  try {
    const content = await fs.readFile(EMBEDDINGS_FILE, 'utf-8');
    const embeddingsData = JSON.parse(content);
    console.log(`Loaded ${embeddingsData.length} embeddings from ${EMBEDDINGS_FILE}`);
    return embeddingsData.map(item => ({
      id: item.id,
      data: { title: item.title, snippet: item.snippet },
      embedding: item.embedding
    }));
  } catch (err) {
    console.warn('No existing embeddings file found. Will compute embeddings.');
    return null;
  }
}

let cachedDocuments = [];

async function initializeDocuments() {
  cachedDocuments = await loadEmbeddingsFromFile();
  if (!cachedDocuments) {
    const documents = await loadDocuments();

    console.log('Embedding documents...');
    const allEmbeddings = await embedDocumentsInBatches(documents);

    allEmbeddings.forEach((embedding, i) => {
      documents[i].embedding = embedding;
    });

    await saveEmbeddingsToFile(documents);
    cachedDocuments = documents;
    console.log('Embeddings ready.');
  }
}

// Initialize on server start
initializeDocuments();

app.post('/generate', async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  try {
    console.log(`Embedding user prompt...`);
    const embedResponse = await cohere.embed({
      texts: [prompt],
      model: 'embed-multilingual-v3.0',
      input_type: 'search_document'
    });

    const queryEmbedding = embedResponse.embeddings[0];

    const topDocuments = getTopKDocuments(queryEmbedding, cachedDocuments, 10);

    console.log(`Retrieved top ${topDocuments.length} documents.`);

    const response = await cohere.chat({
      model: 'command-r-plus',
      message: prompt,
      documents: topDocuments.map(doc => ({
        text: `${doc.data.title}. ${doc.data.snippet}`
      })),
      //CHANGE THIS TO FIT THE CONTEXT OG YOUR APP
      preamble: 'You are a professional and friendly expert travel assistant named Y-TravelBot, working for Y-Travels. You must answer the users questions using ONLY the information provided in the documents below whenever possible. If a topic is not covered by the documents, you may use your own knowledge — but ONLY in the domain of travel and tourism. Stay strictly within this domain: travel, countries, cities, attractions, history, geography, local cuisine, culture, and things to do. DO NOT provide information about politics, economics, safety advice, or unrelated topics. Always write in a helpful, engaging tone suitable for a travel website audience. If country or place not refered to in documents please tell user to click generate itinerary in the nav bar',
      temperature: 0.3
    });

    console.log('Cohere chat response:', JSON.stringify(response, null, 2));

    res.json({
      text: response.text,
      citations: response.citations ?? []
    });

  } catch (err) {
    console.error('Error communicating with Cohere API:', err);
    res.status(500).json({ error: 'Cohere request failed' });
  }
});

app.post('/holiday', async (req, res) => {
  const { userInput } = req.body;

  try {
    const response = await cohere.chat({
      message: `Generate a holiday itinerary based on this request: "${userInput}". Format the response as Day-wise itinerary.`,
      temperature: 0.7,
      max_tokens: 1000
    });

    res.json({ itinerary: response.text });
  } catch (err) {
    console.error('Error:', err);
    res.status(500).json({ error: 'Failed to generate itinerary' });
  }
});

app.listen(5000, () => {
  console.log('Listening on http://localhost:5000');
});
