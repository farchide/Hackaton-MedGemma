#!/usr/bin/env node
/**
 * Claude Flow Auto-Memory Hook (ESM)
 * Handles automatic memory import on session start and sync on stop.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(process.cwd(), '.claude-flow', 'data');
const MEMORY_FILE = path.join(DATA_DIR, 'memory.json');
const SESSION_FILE = path.join(DATA_DIR, 'session-memory.json');

function loadJSON(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
  } catch (e) {
    // Ignore parse errors
  }
  return {};
}

function saveJSON(filePath, data) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

const commands = {
  import() {
    const memory = loadJSON(MEMORY_FILE);
    const session = loadJSON(SESSION_FILE);
    const keys = Object.keys(memory).filter(k => !k.startsWith('_'));

    if (keys.length > 0 || Object.keys(session).length > 0) {
      console.log(`[MEMORY] Imported ${keys.length} memory keys, session context loaded`);
    } else {
      console.log('[MEMORY] No prior memory found, starting fresh');
    }

    // Update session timestamp
    saveJSON(SESSION_FILE, {
      ...session,
      _lastImport: new Date().toISOString(),
      _sessionId: `session-${Date.now()}`,
    });
  },

  sync() {
    const session = loadJSON(SESSION_FILE);
    saveJSON(SESSION_FILE, {
      ...session,
      _lastSync: new Date().toISOString(),
    });
    console.log('[MEMORY] Session memory synced');
  },
};

const [command] = process.argv.slice(2);

if (command && commands[command]) {
  try {
    commands[command]();
  } catch (e) {
    console.log(`[WARN] auto-memory-hook ${command}: ${e.message}`);
  }
} else {
  console.log('Usage: auto-memory-hook.mjs <import|sync>');
}
