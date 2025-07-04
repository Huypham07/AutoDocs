"use client";

import type React from "react";

import { useState, useEffect } from "react";
import { Code, ChevronDown, Download, Loader2 } from "lucide-react";
import { useRouter, useParams } from "next/navigation";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import logo from "@/public/logo.png";
import Image from "next/image";

interface DocumentationItem {
  title: string;
  type: "file" | "section";
  path?: string;
  children?: DocumentationItem[];
}

export default function DocsDetails() {
  const params = useParams();
  const owner = params.owner as string;
  const repoName = params.repo as string;

  const [docMode, setDocMode] = useState<"high-level" | "low-level">("high-level");
  const [docTypeLoading, setDocTypeLoading] = useState(false);
  const [activeSection, setActiveSection] = useState<string>("");

  const router = useRouter();

  const goHome = () => {
    router.push("/");
  };

  // Add scroll tracking with Intersection Observer
  useEffect(() => {
    if (docMode !== "high-level") return;

    const sections = [
      "system-architecture",
      "core-features",
      "home-page",
      "repository-wiki-page",
      "ask-component",
      "configuration-modals",
      "visualization-components",
      "projects-management",
      "rag-system-data-pipeline",
      "chat-completion-api",
      "core-api-endpoints",
      "configuration-system",
      "multi-provider-support",
      "local-ollama-integration",
      "model-configuration",
      "docker-configuration",
      "cicd-pipeline",
      "environment-setup",
      "internationalization",
      "development-setup",
    ];

    const observer = new IntersectionObserver(
      (entries) => {
        const allSections = sections.map((id) => document.getElementById(id)).filter((el) => el !== null);

        const headerOffset = 75;
        let closestSection = null;
        let closestDistance = Infinity;

        allSections.forEach((section) => {
          const rect = section.getBoundingClientRect();
          const distanceFromTop = Math.abs(rect.top - headerOffset);

          if (rect.top <= headerOffset + 50 && rect.bottom >= headerOffset) {
            if (distanceFromTop < closestDistance) {
              closestDistance = distanceFromTop;
              closestSection = section;
            }
          }
        });

        const scrollTop = window.pageYOffset;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        if (scrollTop + windowHeight >= documentHeight - 150) {
          closestSection = allSections[allSections.length - 1];
        }

        if (closestSection) {
          setActiveSection(closestSection.id);
        }
      },
      {
        rootMargin: "-75px 0px -80% 0px",
        threshold: [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
      }
    );

    sections.forEach((sectionId) => {
      const element = document.getElementById(sectionId);
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, [docMode]);

  // Mock documentation data
  const highLevelDocs: DocumentationItem[] = [
    {
      title: "Overview",
      type: "section",
      children: [
        { title: "System Architecture", type: "file", path: "#system-architecture" },
        { title: "Core Features", type: "file", path: "#core-features" },
      ],
    },
    {
      title: "Frontend Components",
      type: "section",
      children: [
        { title: "Home Page", type: "file", path: "#home-page" },
        { title: "Repository Wiki Page", type: "file", path: "#repository-wiki-page" },
        { title: "Ask Component", type: "file", path: "#ask-component" },
        { title: "Configuration & Modals", type: "file", path: "#configuration-modals" },
        { title: "Visualization Components", type: "file", path: "#visualization-components" },
        { title: "Projects Management", type: "file", path: "#projects-management" },
      ],
    },
    {
      title: "Backend Systems",
      type: "section",
      children: [
        { title: "RAG System & Data Pipeline", type: "file", path: "#rag-system-data-pipeline" },
        { title: "Chat Completion API", type: "file", path: "#chat-completion-api" },
        { title: "Core API Endpoints", type: "file", path: "#core-api-endpoints" },
        { title: "Configuration System", type: "file", path: "#configuration-system" },
      ],
    },
    {
      title: "Model Integration",
      type: "section",
      children: [
        { title: "Multi-Provider Support", type: "file", path: "#multi-provider-support" },
        { title: "Local Ollama Integration", type: "file", path: "#local-ollama-integration" },
        { title: "Model Configuration", type: "file", path: "#model-configuration" },
      ],
    },
    {
      title: "Deployment & Infrastructure",
      type: "section",
      children: [
        { title: "Docker Configuration", type: "file", path: "#docker-configuration" },
        { title: "CI/CD Pipeline", type: "file", path: "#cicd-pipeline" },
        { title: "Environment Setup", type: "file", path: "#environment-setup" },
      ],
    },
    {
      title: "Internationalization",
      type: "file",
      path: "#internationalization",
    },
    {
      title: "Development Setup",
      type: "file",
      path: "#development-setup",
    },
  ];

  const handleDocModeChange = async (mode: "high-level" | "low-level") => {
    setDocTypeLoading(true);

    // Simulate loading effect
    await new Promise((resolve) => setTimeout(resolve, 500));

    setDocMode(mode);
    setDocTypeLoading(false);
  };

  const renderDocumentationItem = (item: DocumentationItem, level = 0) => {
    const handleClick = (path?: string) => {
      if (path && path.startsWith("#")) {
        const element = document.querySelector(path);
        if (element) {
          element.scrollIntoView({ behavior: "smooth" });
        }
      }
    };

    const isActive = item.path && activeSection === item.path.substring(1);

    if (item.type === "section" && item.children) {
      const hasActiveChild = item.children.some((child) => child.path && activeSection === child.path.substring(1));

      return (
        <div key={item.title} className="mb-3">
          <div
            className={`p-2 rounded-lg transition-all duration-300 cursor-pointer group ${
              hasActiveChild ? "bg-blue-600 text-white shadow-md" : "bg-gray-100 hover:bg-gray-200"
            }`}
            onClick={() => {
              // Scroll to first child if section is clicked
              if (item.children && item.children[0]?.path) {
                handleClick(item.children[0].path);
              }
            }}>
            <div className="flex items-center justify-between">
              <span className={`font-medium text-sm ${hasActiveChild ? "text-white" : "text-foreground"}`}>
                {item.title}
              </span>
              <div
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  hasActiveChild ? "bg-white" : "bg-gray-400 group-hover:bg-gray-500"
                }`}
              />
            </div>
          </div>
          {/* Render children recursively */}
          <div className="ml-4 mt-2 space-y-1 border-l-2 border-gray-200 pl-4">
            {item.children.map((child) => renderDocumentationItem(child, level + 1))}
          </div>
        </div>
      );
    }

    return (
      <div
        key={item.title}
        className={`flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all duration-300 text-sm group ${
          isActive
            ? "bg-blue-100 text-blue-800 border-l-3 border-blue-500 shadow-sm"
            : "text-muted-foreground hover:text-gray-700 hover:bg-gray-50"
        }`}
        onClick={() => handleClick(item.path)}>
        <div
          className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${
            isActive ? "bg-blue-500" : "bg-gray-300 group-hover:bg-gray-400"
          }`}
        />
        <span className="flex-1">{item.title}</span>
      </div>
    );
  };

  const exportAsMarkdown = () => handleExport("md");
  const exportAsPDF = () => handleExport("pdf");
  const exportAsDocs = () => handleExport("docx");

  const handleExport = (format: "docx" | "pdf" | "md") => {
    toast("This feature is currently under development.", {
      description: `Exporting to ${format.toUpperCase()} will be available soon.`,
      action: {
        label: "Close",
        onClick: () => toast.dismiss(),
      },
      duration: 3000,
      className: "text-red-600",
      actionButtonStyle: { backgroundColor: "ButtonShadow", color: "black" },
      position: "top-center",
      descriptionClassName: "text-gray-700",
      style: {
        backgroundColor: "white",
        outline: "1px solid #ccc",
      },
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="cursor-pointer flex items-center gap-3" onClick={goHome}>
                  <Image src={logo} alt="AutoDocs Logo" width={32} height={32} className="rounded-full" />
                  <span className="text-lg font-medium">AutoDocs</span>
                </div>
                <div className="w-[1px] h-9 bg-slate-400"></div>
              </div>
            </div>

            <div className="w-full">
              <div className="container mx-auto">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-base font- text-foreground">
                      {owner}/{repoName}
                    </span>
                    <div className="text-xs text-muted-foreground">Last generated: 06/06/2004</div>
                  </div>

                  <Select onValueChange={handleDocModeChange}>
                    <SelectTrigger>
                      {docTypeLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Loading...
                        </>
                      ) : (
                        <>{docMode === "high-level" ? "High-Level Documentation" : "Low-Level Documentation"}</>
                      )}
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="high-level">High-Level Documentation</SelectItem>
                      <SelectItem value="low-level">Low-Level Documentation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="relative">
                <DropdownMenu>
                  <DropdownMenuTrigger>
                    <div className="bg-blue-500 text-white inline-flex items-center justify-center gap-2 whitespace-nowrap h-9 px-3 rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input hover:bg-blue-600">
                      <Download color="white" className="w-4 h-4 mr-2" />
                      Export
                      <ChevronDown color="white" className="w-4 h-4 ml-2" />
                    </div>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem className="cursor-pointer" onClick={exportAsMarkdown}>
                      Export as Markdown
                    </DropdownMenuItem>
                    <DropdownMenuItem className="cursor-pointer" onClick={exportAsPDF}>
                      Export as PDF
                    </DropdownMenuItem>
                    <DropdownMenuItem className="cursor-pointer" onClick={exportAsDocs}>
                      Export as Docs
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Pane - Content */}
          <div className="lg:col-span-4">
            <div className="prose max-w-none">
              {docMode === "high-level" ? (
                <>
                  <div className="mb-6">
                    <h1 className="text-3xl font-bold">Documentation Overview</h1>
                  </div>
                  <div id="system-architecture" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">System Architecture</h3>
                    <p>This section covers the overall system architecture and design patterns used in.</p>
                    <p>
                      The architecture follows modern best practices with clear separation of concerns, scalable design
                      patterns, and maintainable code structure. The system is built with modularity in mind, allowing
                      for easy extension and modification of individual components without affecting the overall system
                      stability.
                    </p>
                  </div>

                  <div id="core-features" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Core Features</h3>
                    <p>Key features and capabilities of the system.</p>
                    <p>
                      The system provides a comprehensive set of features designed to meet modern application
                      requirements. These include user authentication, data management, real-time updates, and
                      integration capabilities with external services.
                    </p>
                  </div>

                  <div id="home-page" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Home Page</h3>
                    <p>Documentation for the main landing page component.</p>
                    <p>
                      The home page serves as the primary entry point for users, providing an intuitive interface that
                      guides users through the application's main features and functionality.
                    </p>
                  </div>

                  <div id="repository-wiki-page" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Repository Wiki Page</h3>
                    <p>Details about the repository wiki functionality.</p>
                    <p>
                      The wiki page provides comprehensive documentation and knowledge base functionality, allowing
                      users to create, edit, and organize documentation in a collaborative environment.
                    </p>
                  </div>

                  <div id="ask-component" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Ask Component</h3>
                    <p>Interactive question and answer component documentation.</p>
                    <p>
                      This component enables users to ask questions and receive intelligent responses, leveraging
                      advanced AI capabilities to provide accurate and contextual information.
                    </p>
                  </div>

                  <div id="configuration-modals" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Configuration & Modals</h3>
                    <p>Settings and modal dialog components.</p>
                    <p>
                      The configuration system provides users with flexible options to customize their experience,
                      including theme preferences, notification settings, and application behavior.
                    </p>
                  </div>

                  <div id="visualization-components" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Visualization Components</h3>
                    <p>Charts, graphs, and data visualization components.</p>
                    <p>
                      These components provide rich data visualization capabilities, including interactive charts,
                      graphs, and dashboards that help users understand complex data relationships.
                    </p>
                  </div>

                  <div id="projects-management" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Projects Management</h3>
                    <p>Project organization and management features.</p>
                    <p>
                      The project management system allows users to organize their work, track progress, and collaborate
                      effectively with team members across multiple projects.
                    </p>
                  </div>

                  <div id="rag-system-data-pipeline" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">RAG System & Data Pipeline</h3>
                    <p>Retrieval-Augmented Generation system and data processing pipeline.</p>
                    <p>
                      The RAG system combines retrieval and generation capabilities to provide accurate, contextual
                      responses based on a comprehensive knowledge base and real-time data processing.
                    </p>
                  </div>

                  <div id="chat-completion-api" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Chat Completion API</h3>
                    <p>API endpoints for chat and completion functionality.</p>
                    <p>
                      This API provides robust chat completion capabilities, supporting various conversation patterns
                      and intelligent response generation for enhanced user interactions.
                    </p>
                  </div>

                  <div id="core-api-endpoints" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Core API Endpoints</h3>
                    <p>Essential API endpoints and their documentation.</p>
                    <p>
                      The core API provides fundamental operations including data retrieval, user management,
                      authentication, and system configuration through well-documented RESTful endpoints.
                    </p>
                  </div>

                  <div id="configuration-system" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Configuration System</h3>
                    <p>System configuration and settings management.</p>
                    <p>
                      The configuration system provides centralized management of application settings, environment
                      variables, and runtime parameters for optimal system performance.
                    </p>
                  </div>

                  <div id="multi-provider-support" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Multi-Provider Support</h3>
                    <p>Support for multiple AI model providers.</p>
                    <p>
                      The system supports integration with various AI model providers, allowing users to choose the best
                      model for their specific use cases and requirements.
                    </p>
                  </div>

                  <div id="local-ollama-integration" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Local Ollama Integration</h3>
                    <p>Integration with local Ollama models.</p>
                    <p>
                      Local Ollama integration provides privacy-focused AI capabilities by running models locally,
                      ensuring data security and reducing dependency on external services.
                    </p>
                  </div>

                  <div id="model-configuration" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Model Configuration</h3>
                    <p>AI model setup and configuration options.</p>
                    <p>
                      Comprehensive model configuration options allow fine-tuning of AI behavior, performance
                      parameters, and output characteristics to match specific application requirements.
                    </p>
                  </div>

                  <div id="docker-configuration" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Docker Configuration</h3>
                    <p>Containerization setup and Docker configuration.</p>
                    <p>
                      Docker configuration provides consistent deployment environments, simplified scaling, and reliable
                      application distribution across different platforms and environments.
                    </p>
                  </div>

                  <div id="cicd-pipeline" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">CI/CD Pipeline</h3>
                    <p>Continuous integration and deployment setup.</p>
                    <p>
                      The CI/CD pipeline automates testing, building, and deployment processes, ensuring code quality
                      and enabling rapid, reliable software delivery.
                    </p>
                  </div>

                  <div id="environment-setup" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Environment Setup</h3>
                    <p>Development and production environment configuration.</p>
                    <p>
                      Detailed environment setup instructions ensure consistent development experiences and reliable
                      production deployments across different systems and configurations.
                    </p>
                  </div>

                  <div id="internationalization" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Internationalization</h3>
                    <p>Multi-language support and localization.</p>
                    <p>
                      Comprehensive internationalization support enables the application to serve users in multiple
                      languages and regions, with proper localization of content and user interface elements.
                    </p>
                  </div>

                  <div id="development-setup" className="scroll-mt-32 mb-4">
                    <h3 className="text-2xl font-bold">Development Setup</h3>
                    <p>Getting started with development environment setup.</p>
                    <p>
                      Step-by-step development setup guide helps new contributors quickly establish a working
                      development environment and begin contributing to the project effectively.
                    </p>
                  </div>
                </>
              ) : (
                <div className="flex items-center justify-center h-96">
                  <div className="text-center">
                    <div className="text-6xl mb-4">🚧</div>
                    <h3 className="text-xl font-semibold mb-2">This section is under development</h3>
                    <p className="text-muted-foreground">Low-level code documentation will be available soon.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Pane - TOC */}
          <div className="lg:col-span-1">
            <div className="sticky top-[73px] h-fit">
              <div className="space-y-2">
                {docMode === "high-level" ? (
                  <>{highLevelDocs.map((item) => renderDocumentationItem(item))}</>
                ) : (
                  <div className="text-center text-muted-foreground py-8">
                    <Code className="w-8 h-8 mx-auto mb-2" />
                    <p className="text-sm">Code documentation coming soon</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
