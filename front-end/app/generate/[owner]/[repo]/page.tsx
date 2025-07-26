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
import { DocumantationResponse, Section, Page } from "@/schemas/task.schema";
import Markdown from "@/components/Markdown";

export default function DocsDetails() {
  const params = useParams();
  const owner = params.owner as string;
  const repoName = params.repo as string;

  const [docMode, setDocMode] = useState<"view" | "ask">("view");
  const [docTypeLoading, setDocTypeLoading] = useState(false);
  const [activeSection, setActiveSection] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [documentation, setDocumentation] = useState<DocumantationResponse | null>(null);
  const [allSectionIds, setAllSectionIds] = useState<string[]>([]);

  const router = useRouter();

  const goHome = () => {
    router.push("/");
  };

  // Collect all page IDs for scroll tracking
  const collectPageIds = (sections: Section[]): string[] => {
    const ids: string[] = [];

    const processSection = (section: Section) => {
      section.pages.sort((a, b) => a.page_id.localeCompare(b.page_id)).forEach((page) => ids.push(page.page_id));
      section.subsections
        .sort((a, b) => a.section_id.localeCompare(b.section_id))
        .forEach((subsection) => processSection(subsection));
    };

    sections.sort((a, b) => a.section_id.localeCompare(b.section_id)).forEach((section) => processSection(section));
    return ids;
  };

  // Scroll tracking with Intersection Observer
  useEffect(() => {
    if (docMode !== "view" || allSectionIds.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleSections = allSectionIds.map((id) => document.getElementById(id)).filter((el) => el !== null);

        const headerOffset = 75;
        let closestSection = null;
        let closestDistance = Infinity;

        visibleSections.forEach((section) => {
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
          closestSection = visibleSections[visibleSections.length - 1];
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

    allSectionIds.forEach((sectionId) => {
      const element = document.getElementById(sectionId);
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, [docMode, allSectionIds]);

  const handleDocModeChange = async (mode: "view" | "ask") => {
    setDocTypeLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 500));
    setDocMode(mode);
    setDocTypeLoading(false);
  };

  const renderDocumentationItem = (section: Section, level = 0) => {
    const handleClick = (id: string) => {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    };

    // Check if any page in this section or subsections is active
    const isAnyPageActive = (sec: Section): boolean => {
      if (sec.pages.some((page) => activeSection === page.page_id)) {
        return true;
      }
      return sec.subsections.some((subsec) => isAnyPageActive(subsec));
    };

    const hasActiveChild = isAnyPageActive(section);

    return (
      <div key={section.section_id} className="mb-3">
        <div
          className={`p-2 rounded-lg transition-all duration-300 cursor-pointer group ${
            hasActiveChild ? "bg-blue-600 text-white shadow-md" : "bg-gray-100 hover:bg-gray-200"
          }`}
          onClick={() => {
            // Scroll to first page if section is clicked
            if (section.pages.length > 0) {
              handleClick(section.pages[0].page_id);
            }
          }}>
          <div className="flex items-center justify-between">
            <span className={`font-medium text-sm ${hasActiveChild ? "text-white" : "text-foreground"}`}>
              {section.section_title}
            </span>
            <div
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                hasActiveChild ? "bg-white" : "bg-gray-400 group-hover:bg-gray-500"
              }`}
            />
          </div>
        </div>

        {/* Render pages */}
        {section.pages.length > 0 && (
          <div className="ml-4 mt-2 space-y-1 border-l-2 border-gray-200 pl-4">
            {section.pages.map((page) => {
              const isActive = activeSection === page.page_id;
              return (
                <div
                  key={page.page_id}
                  className={`flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all duration-300 text-sm group ${
                    isActive
                      ? "bg-blue-100 text-blue-800 border-l-3 border-blue-500 shadow-sm"
                      : "text-muted-foreground hover:text-gray-700 hover:bg-gray-50"
                  }`}
                  onClick={() => handleClick(page.page_id)}>
                  <div
                    className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${
                      isActive ? "bg-blue-500" : "bg-gray-300 group-hover:bg-gray-400"
                    }`}
                  />
                  <span className="flex-1">{page.page_title}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Render subsections recursively */}
        {section.subsections.length > 0 && (
          <div className="ml-4 mt-2 space-y-1 border-l-2 border-gray-200 pl-4">
            {section.subsections.map((subsection) => renderDocumentationItem(subsection, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const renderContent = () => {
    if (!documentation) return null;

    const renderSection = (section: Section) => (
      <div key={section.section_id} className="mb-8">
        <h2 className="text-2xl font-bold mb-4">{section.section_title}</h2>

        {/* Render pages */}
        {section.pages.map((page) => (
          <div key={page.page_id} id={page.page_id} className="scroll-mt-32 mb-6">
            <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
              <Markdown content={page.content} />
            </div>
            {page.file_paths.length > 0 && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <h4 className="text-sm font-medium mb-2">Related Files:</h4>
                <ul className="text-sm text-gray-600">
                  {page.file_paths.map((path, index) => (
                    <li key={index} className="font-mono">
                      {path}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        {/* Render subsections recursively */}
        {section.subsections.map((subsection) => renderSection(subsection))}
      </div>
    );

    return (
      <div className="prose max-w-none">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">{documentation.title}</h1>
          <p className="text-lg text-gray-600 mt-2">{documentation.description}</p>
        </div>
        {documentation.root_sections
          .sort((a, b) => a.section_id.localeCompare(b.section_id))
          .map((section) => renderSection(section))}
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

  useEffect(() => {
    const fetchDocumentation = async () => {
      try {
        const response = await fetch(`/api/documents/${owner}/${repoName}`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const data: DocumantationResponse = await response.json();
          setDocumentation(data);
          const pageIds = collectPageIds(data.root_sections);
          setAllSectionIds(pageIds);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred";
        toast(`Failed to fetch documentation: ${errorMessage}`, {
          action: {
            label: "Close",
            onClick: () => toast.dismiss(),
          },
          duration: 3000,
          className: "text-red-600",
          actionButtonStyle: { backgroundColor: "ButtonShadow", color: "black" },
          position: "top-center",
          style: {
            backgroundColor: "white",
            outline: "1px solid #ccc",
          },
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDocumentation();
  }, [owner, repoName]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p>Loading documentation...</p>
        </div>
      </div>
    );
  }

  if (!documentation) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h3 className="text-xl font-semibold mb-2">Documentation not found</h3>
          <p className="text-muted-foreground">Unable to load documentation for this repository.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="cursor-pointer flex items-center gap-3" onClick={goHome}>
                  <span className="w-8 h-8">
                    <Image src={logo} alt="AutoDocs Logo" width={32} height={32} className="rounded-full" />
                  </span>
                  <span className="text-lg font-medium">AutoDocs</span>
                </div>
                <div className="w-[1px] h-9 bg-slate-400"></div>
              </div>
            </div>

            <div className="w-full">
              <div className="container mx-auto">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-base font-medium text-foreground">
                      {documentation.owner}/{documentation.repo_name}
                    </span>
                    <div className="text-xs text-muted-foreground">
                      Last generated: {formatDate(documentation.updated_at)}
                    </div>
                  </div>

                  <Select onValueChange={handleDocModeChange}>
                    <SelectTrigger>
                      {docTypeLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Loading...
                        </>
                      ) : (
                        <>{docMode === "view" ? "View Documentation" : "Ask about Documentation"}</>
                      )}
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="view">View Documentation</SelectItem>
                      <SelectItem value="ask">Ask about Documentation</SelectItem>
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
            {docMode === "view" ? (
              renderContent()
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

          {/* Right Pane - TOC */}
          <div className="lg:col-span-1">
            <div className="sticky top-[73px] h-fit">
              <div className="space-y-2">
                {docMode === "view" && documentation ? (
                  <>
                    {documentation.root_sections
                      .sort((a, b) => a.section_id.localeCompare(b.section_id))
                      .map((section) => renderDocumentationItem(section))}
                  </>
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
