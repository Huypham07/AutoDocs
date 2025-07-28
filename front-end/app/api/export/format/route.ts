import { NextRequest, NextResponse } from "next/server";

const SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json();

    const targetUrl = `${SERVER_BASE_URL}/export/format`;

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.text();
    return new NextResponse(result, {
      status: response.status,
    });
  } catch (error) {
    console.error("Error in API proxy route:", error);
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204, // No Content
    headers: {
      "Access-Control-Allow-Origin": "*", // Be more specific in production if needed
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization", // Adjust as per client's request headers
    },
  });
}
