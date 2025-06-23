import { NextRequest, NextResponse } from "next/server";

const SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json();

    const targetUrl = `${SERVER_BASE_URL}/streaming/completion`;

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      const errorHeaders = new Headers();
      response.headers.forEach((value, key) => {
        errorHeaders.set(key, value);
      });
      return new NextResponse(errorText, {
        status: response.status,
        statusText: response.statusText,
        headers: errorHeaders,
      });
    }

    if (!response.body) {
      return new NextResponse("No response body from target server", {
        status: 500,
      });
    }

    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
        } catch (error) {
          console.error("Error reading from response body:", error);
          controller.error(error);
        } finally {
          reader.releaseLock();
          controller.close();
        }
      },
      cancel(reason) {
        console.log("Stream cancelled:", reason);
      },
    });

    const headers = new Headers(response.headers);
    const contentType = response.headers.get("Content-Type");
    if (contentType) {
      headers.set("Content-Type", contentType);
    }
    headers.set("Cache-Control", "no-cache, no-transform");

    return new NextResponse(stream, {
      status: response.status,
      headers: headers,
    });
  } catch (error) {
    console.error("Error in streaming completion route:", error);
    return new NextResponse("Internal Server Error", {
      status: 500,
      statusText: "Internal Server Error",
      headers: {
        "Content-Type": "text/plain",
      },
    });
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Accept",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}
