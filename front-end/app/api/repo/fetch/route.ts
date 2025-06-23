import { NextRequest, NextResponse } from "next/server";

const SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json();

    const targetUrl = `${SERVER_BASE_URL}/repo/fetch`;

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      return NextResponse.json({ error: `Backend server returned ${response.status}` }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error forwarding request to backend:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
