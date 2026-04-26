import re

class SentenceAwareBuffer:
    def __init__(self, weight_threshold=1.5):
        self.text_buffer = ""
        self.pending_weights = []
        self.weight_threshold = weight_threshold

    def process_chunk(self, new_text, chunk_weight):
        """
        new_text: ASR output from a 30s chunk
        chunk_weight: The max weight from the Emotion Engine for this period
        """
        self.text_buffer += " " + new_text
        self.pending_weights.append(chunk_weight)

        # Regex to find complete sentences (ending in . ! or ?)
        sentences = re.split(r'(?<=[.!?])\s+', self.text_buffer)

        # If the last item in 'sentences' doesn't end in punctuation, 
        # it's an incomplete sentence. Keep it in the buffer.
        if not re.search(r'[.!?]$', sentences[-1]):
            self.text_buffer = sentences.pop() 
        else:
            self.text_buffer = ""

        ready_to_vectorize = []
        if sentences:
            # Average the weights of the chunks that formed these sentences
            final_weight = max(self.pending_weights) if self.pending_weights else 1.0
            
            for sentence in sentences:
                if len(sentence.strip()) > 10: # Avoid noise/empty strings
                    ready_to_vectorize.append({
                        "text": sentence.strip(),
                        "weight": final_weight
                    })
            
            # Reset weights after processing
            self.pending_weights = []
            
        return ready_to_vectorize

# Example Usage in your main loop:
# buffer = SentenceAwareBuffer()
# results = buffer.process_chunk("This is a critical point", 2.0)
# results += buffer.process_chunk(" about the self-attention mechanism.", 1.2)
# # Output: [{'text': 'This is a critical point about the self-attention mechanism.', 'weight': 2.0}]