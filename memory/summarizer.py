from anthropic import Anthropic

class ConversationSummarizer:
    """Summarizes conversation history to save tokens and disk space."""
    
    def __init__(self, store, client=None):
        self.store = store
        self.client = client or Anthropic()
    
    def summarize_and_prune(self, max_turns_before_summary: int = 50):
        """Summarize old turns and delete the raw data."""
        turns = self.store.get_recent_turns(n=max_turns_before_summary)
        if len(turns) < max_turns_before_summary:
            return  # Not enough to summarize yet
        
        # Use Haiku — fast and cheap for summarization
        conversation_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in turns
        )
        
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system="Summarize this conversation concisely. Focus on: facts learned about the user, "
                   "decisions made, preferences expressed, and any ongoing topics. Be brief.",
            messages=[{"role": "user", "content": conversation_text}],
        )
        
        summary = response.content[0].text
        
        # Get the ID range to prune (you'd need to track IDs — simplified here)
        self.store.save_summary(summary, start_id=0, end_id=len(turns))
