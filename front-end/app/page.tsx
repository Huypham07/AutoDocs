"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Github, Search, AlertCircle, Loader2, Lock, Code } from "lucide-react";
import { useRouter } from "next/navigation";
import { TaskBody, TaskResponse } from "@/schemas/task.schema";

export default function AutoDocs() {
  const [repoUrl, setRepoUrl] = useState("https://github.com/Huypham07/AutoDocs");
  const [accessToken, setAccessToken] = useState("");
  const [showAccessToken, setShowAccessToken] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [streamingContent, setStreamingContent] = useState("");

  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    setStreamingContent("");

    try {
      // get base URL from env
      const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || "localhost:8000";

      const requestBody: TaskBody = {
        repo_url: repoUrl,
        ...(accessToken && { access_token: accessToken }),
      };

      const response = await fetch(`/api/documents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || `Error fetch repo: ${response.status}`;

        // Handle specific 4xx errors
        if (response.status === 401) {
          setShowAccessToken(true);
          setError("Repository requires authentication. Please provide a valid access token below.");
          return;
        } else if (response.status === 403) {
          setShowAccessToken(true);
          setError("Access denied. Please check your access token permissions.");
          return;
        } else if (response.status === 404) {
          setError(
            "Repository not found. Please check the repository URL or provide a valid access token if it's a private repository."
          );
          setShowAccessToken(true);
          return;
        }

        throw new Error(errorMessage);
      }

      // Process the response
      let content = "";
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          content += chunk;

          // Update streaming responses - only keep last 2 lines
          setStreamingContent((prev) => prev + chunk);
        }
        // Ensure final decoding
        content += decoder.decode();
      } catch (readError) {
        console.error("Error reading stream:", readError);
        throw new Error("Error processing response stream");
      }

      // Redirect to the documentation page with owner and repo as query parameters
      // router.push(`/generate/${owner}/${repo_name}`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred";
      setError(errorMessage);

      // Show access token input for auth-related errors
      if (
        errorMessage.toLowerCase().includes("authentication") ||
        errorMessage.toLowerCase().includes("access") ||
        errorMessage.toLowerCase().includes("token")
      ) {
        setShowAccessToken(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setError("");
    setShowAccessToken(false);
    setAccessToken("");
    setStreamingContent("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Github className="w-8 h-8 text-blue-600" />
            <h1 className="text-2xl font-bold">AutoDocs</h1>
          </div>
          <CardTitle>Analyze Git Repository</CardTitle>
          <CardDescription>
            Enter a GitHub/GitLab repository URL to generate comprehensive documentation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="repo-url" className="flex items-center gap-2">
                <Code className="w-4 h-4" />
                Repository URL
              </Label>
              <Input
                id="repo-url"
                type="url"
                placeholder="https://github.com/username/repository"
                value={repoUrl}
                onChange={(e) => {
                  setRepoUrl(e.target.value);
                  resetForm();
                }}
                disabled={loading}
                className="w-full"
              />
            </div>

            {showAccessToken && (
              <div className="space-y-2">
                <Label htmlFor="access-token" className="flex items-center gap-2">
                  <Lock className="w-4 h-4" />
                  Access Token
                </Label>
                <Input
                  id="access-token"
                  type="password"
                  placeholder="Your GitHub/GitLab access token"
                  value={accessToken}
                  onChange={(e) => setAccessToken(e.target.value)}
                  disabled={loading}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Required for private repositories or when rate limits are exceeded
                </p>
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={loading || !repoUrl.trim()}>
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing Repository...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Analyze Repository
                </>
              )}
            </Button>
          </form>

          {/* Streaming Response Display */}
          {streamingContent && (
            <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <div className="space-y-1">
                {(() => {
                  // Split content into lines and get last 2 lines
                  const lines = streamingContent.split("\n").filter((line) => line.trim());
                  const lastTwoLines = lines.slice(-2);

                  return lastTwoLines.map((line, index) =>
                    line.includes("Error:") ? (
                      <div
                        key={`${lines.length}-${index}`}
                        className={`text-sm transition-opacity duration-500 ${
                          index === lastTwoLines.length - 1
                            ? "text-red-600 opacity-80 text-sm"
                            : "text-red-400 opacity-50 text-xs"
                        }`}>
                        {line.trim()}
                      </div>
                    ) : (
                      <div
                        key={`${lines.length}-${index}`}
                        className={`transition-opacity duration-500 overflow-x-hidden whitespace-nowrap truncate overflow-ellipsis${
                          index === lastTwoLines.length - 1
                            ? "text-gray-600 opacity-80 text-sm"
                            : "text-gray-400 opacity-50 text-xs"
                        }`}>
                        {line.trim()}
                      </div>
                    )
                  );
                })()}
              </div>
            </div>
          )}

          {showAccessToken && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>How to get an access token:</strong>
              </p>
              <ul className="text-xs text-blue-700 mt-1 space-y-1">
                <li>• GitHub: Settings → Developer settings → Personal access tokens</li>
                <li>• GitLab: User Settings → Access Tokens</li>
                <li>• Make sure to grant repository read permissions</li>
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
