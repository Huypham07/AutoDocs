"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import xxhash from 'xxhash-wasm';
import { Github, Search, AlertCircle, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";

export default function AutoDocs() {
  const [repoUrl, setRepoUrl] = useState("https://github.com/Huypham07/AutoDocs");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const router = useRouter();

  const validateGitHubUrl = (url: string): { owner: string; repo: string } | null => {
    const githubRegex = /^https?:\/\/github\.com\/([^/]+)\/([^/]+)(?:\/.*)?$/;
    const match = url.match(githubRegex);
    if (match) {
      return { owner: match[1], repo: match[2] };
    }
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const parsed = validateGitHubUrl(repoUrl);
      if (!parsed) {
        throw new Error("Please enter a valid GitHub repository URL");
      }

      // Check if repository exists and is public
      const response = await fetch(`https://api.github.com/repos/${parsed.owner}/${parsed.repo}`);

      if (response.status === 404) {
        throw new Error("Repository not found or is private. Please ensure the repository is public.");
      }

      if (!response.ok) {
        throw new Error("Failed to access repository. Please check the URL and try again.");
      }

      const repoData = await response.json();

      if (repoData.private) {
        throw new Error("This repository is private. Please use a public repository.");
      }

      const hasher = await xxhash();
      const hash = hasher.h64ToString(repoData.full_name);

      router.push(`/docs/${hash}`);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Github className="w-8 h-8 text-blue-600" />
            <h1 className="text-2xl font-bold">AutoDocs</h1>
          </div>
          <CardTitle>Analyze GitHub Repository</CardTitle>
          <CardDescription>Enter a GitHub repository URL to generate comprehensive documentation</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Input
                type="url"
                placeholder="https://github.com/username/repository"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                disabled={loading}
                className="w-full"
              />
            </div>

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
        </CardContent>
      </Card>
    </div>
  );
}
