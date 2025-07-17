import { NextRequest, NextResponse } from "next/server";

const SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json();

    const targetUrl = `${SERVER_BASE_URL}/generate/docs`;

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      const errorHeaders = new Headers();
      response.headers.forEach((value, key) => {
        errorHeaders.set(key, value);
      });
      return new NextResponse(errorBody, {
        status: response.status,
        statusText: response.statusText,
        headers: errorHeaders,
      });
    }

    if (!response.body) {
      return new NextResponse("Stream body from backend is null", { status: 500 });
    }
    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            controller.enqueue(value);
          }
        } catch (error) {
          console.error("Error reading from backend stream in proxy:", error);
          controller.error(error);
        } finally {
          controller.close();
          reader.releaseLock(); // Important to release the lock on the reader
        }
      },
      cancel(reason) {
        console.log("Client cancelled stream request:", reason);
      },
    });

    // Set up headers for the response to the client
    const responseHeaders = new Headers();
    // Copy the Content-Type from the backend response (e.g., 'text/event-stream')
    const contentType = response.headers.get("Content-Type");
    if (contentType) {
      responseHeaders.set("Content-Type", contentType);
    }
    // It's good practice for streams not to be cached or transformed by intermediaries.
    responseHeaders.set("Cache-Control", "no-cache, no-transform");

    return new NextResponse(stream, {
      status: response.status, // Should be 200 for a successful stream start
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("Error in API proxy route:", error);
    let errorMessage = "Internal Server Error in proxy";
    if (error instanceof Error) {
      errorMessage = error.message;
    }
    return new NextResponse(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
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
