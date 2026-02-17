#!/usr/bin/env node
/**
 * Claude Flow Status Line
 * Outputs a brief status string for the Claude Code status bar.
 */

const fs = require('fs');
const path = require('path');

const SESSION_FILE = path.join(process.cwd(), '.claude-flow', 'data', 'session-memory.json');

function getStatus() {
  try {
    if (fs.existsSync(SESSION_FILE)) {
      const session = JSON.parse(fs.readFileSync(SESSION_FILE, 'utf-8'));
      const id = session._sessionId || 'unknown';
      return `claude-flow v3 | ${id}`;
    }
  } catch (e) {
    // Ignore
  }
  return 'claude-flow v3';
}

console.log(getStatus());
