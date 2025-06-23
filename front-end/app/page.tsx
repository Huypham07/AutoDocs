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
import { RepoFetchBody } from "@/schemas/repo.schema";

export default function AutoDocs() {
  const [repoUrl, setRepoUrl] = useState("https://github.com/Huypham07/AutoDocs");
  const [accessToken, setAccessToken] = useState("");
  const [showAccessToken, setShowAccessToken] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // get base URL from env
      const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || "localhost:8000";

      const requestBody: RepoFetchBody = {
        repo_url: repoUrl,
        ...(accessToken && { access_token: accessToken }),
      };

      const response = await fetch(`/api/repo/fetch`, {
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

      const responseData = await response.json();
      const hash_id = responseData.hash_id;

      router.push(`/docs/${hash_id}`);
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
