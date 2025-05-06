
import { useState } from "react";
import { SendIcon, MessageCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

const ChatSection = () => {
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<{role: "user" | "assistant"; content: string}[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      return;
    }

    // Add user message to chat history
    const userMessage = { role: "user" as const, content: query };
    setChatHistory((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from API");
      }

      const data = await response.json();
      
      // Add assistant response to chat history
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant" as const, content: data.response },
      ]);
    } catch (error) {
      console.error("Error fetching chat response:", error);
      toast({
        title: "Error",
        description: "Failed to get a response. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setQuery("");
    }
  };

  return (
    <div className="flex flex-col h-full">
      <Card className="flex-1 mb-4 overflow-hidden flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            <span>AI Assistant</span>
          </CardTitle>
          <CardDescription>
            Ask questions about demand forecasting and clustering models
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 overflow-y-auto pt-0 pb-2">
          <div className="space-y-4">
            {chatHistory.length === 0 ? (
              <div className="text-center p-8 text-muted-foreground">
                <p className="mb-2">No messages yet</p>
                <p className="text-sm">Ask a question about demand forecasting or data clustering</p>
              </div>
            ) : (
              chatHistory.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] px-4 py-2 rounded-lg ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] px-4 py-2 rounded-lg bg-secondary text-secondary-foreground">
                  <Loader2 className="h-5 w-5 animate-spin" />
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <Textarea
          value={query}
          onChange={handleQueryChange}
          placeholder="Ask a question about demand forecasting..."
          className="flex-1 min-h-[60px] resize-none"
        />
        <Button type="submit" size="icon" disabled={isLoading || !query.trim()}>
          {isLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <SendIcon className="h-5 w-5" />
          )}
        </Button>
      </form>
    </div>
  );
};

export default ChatSection;
