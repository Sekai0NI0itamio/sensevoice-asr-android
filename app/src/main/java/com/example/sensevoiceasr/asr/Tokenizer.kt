package com.example.sensevoiceasr.asr

import android.content.Context
import android.util.Log
import org.json.JSONArray

/**
 * Simple tokenizer for SenseVoiceSmall.
 * Loads token mapping from tokens.json and decodes token IDs.
 * Strips all special <|...|> meta tokens (language, emotion, event markers, etc.)
 * leaving only the actual transcribed content.
 */
class Tokenizer(private val context: Context) {
    companion object {
        private const val TAG = "Tokenizer"
        private const val TOKENS_FILE = "tokens.json"
        private val SPECIAL_TOKEN_REGEX = Regex("<\\|[^|]+\\|>")
    }

    private var idToToken: Map<Int, String> = emptyMap()

    fun load(): Boolean {
        return try {
            val json = context.assets.open(TOKENS_FILE).bufferedReader().use { it.readText() }
            val tokensArr = JSONArray(json)
            val map = mutableMapOf<Int, String>()
            for (i in 0 until tokensArr.length()) {
                map[i] = tokensArr.getString(i)
            }
            idToToken = map
            Log.i(TAG, "Loaded ${idToToken.size} tokens")
            true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load tokens.json: ${e.message}. Will use raw output.")
            false
        }
    }

    /**
     * Decode a list of token IDs into text.
     * Uses greedy CTC-style decoding: collapse consecutive duplicates, remove blanks.
     * All special meta-tokens matching <|...|> are filtered out so only
     * the actual spoken content appears in the result.
     */
    fun decode(tokenIds: IntArray): String {
        if (idToToken.isNotEmpty()) {
            val decoded = mutableListOf<String>()
            var prev = -1
            for (id in tokenIds) {
                if (id != prev && id != 0) {
                    idToToken[id]?.let { token ->
                        if (!token.matches(SPECIAL_TOKEN_REGEX)) {
                            decoded.add(token)
                        }
                    }
                }
                prev = id
            }
            return decoded.joinToString("")
                .replace("▁", " ")
                .replace(SPECIAL_TOKEN_REGEX, "")
                .trim()
        }
        return "tokens: ${tokenIds.take(20).joinToString()}"
    }
}