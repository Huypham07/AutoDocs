# AutoDocs Frontend

The frontend application for AutoDocs provides an intuitive web interface for repository analysis and documentation generation.

## Features

### User Interface
- Clean, responsive design built with Next.js and Tailwind CSS
- Real-time progress tracking during repository analysis
- Interactive chat interface for documentation queries
- Repository input with GitHub integration support

### Core Functionality
- Repository URL input and validation
- GitHub access token management for private repositories
- Live analysis progress monitoring with WebSocket integration
- Documentation viewing and export capabilities
- Interactive chat for asking questions about the analyzed codebase

### Visualization
- Architecture overview display
- Module and cluster relationship visualization
- Quality metrics and analysis results presentation
- Responsive layout for desktop and mobile devices

## Quick Setup

### Prerequisites
- Node.js 16.0 or higher
- npm or yarn package manager

### Step 1: Install Dependencies

```bash
npm install
# or
yarn install
```

### Step 2: Environment Configuration

Create a `.env.local` file (if needed for custom backend URL):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Step 3: Start Development Server

```bash
npm run dev
# or
yarn dev
```

The application will be available at http://localhost:3000

## Development

### Project Structure
```
app/                    # Next.js app directory
├── api/               # API route handlers
├── generate/          # Repository analysis pages
└── globals.css        # Global styles

components/            # React components
├── ui/               # Reusable UI components
├── chat-interface.tsx # Chat functionality
├── Markdown.tsx      # Markdown rendering
└── Mermaid.tsx       # Diagram rendering

lib/                  # Utilities and configurations
types/               # TypeScript type definitions
utils/               # Helper functions
```

### Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Linting
npm run lint

# Type checking
npm run type-check
```

### Technology Stack
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI primitives
- **State Management**: React hooks and context
- **WebSocket**: Real-time communication with backend
- **Markdown**: Rich text rendering for documentation
- **Diagrams**: Mermaid.js integration for visualizations

## Building for Production

### Step 1: Build the Application

```bash
npm run build
```

### Step 2: Start Production Server

```bash
npm start
```

### Step 3: Deploy

The application can be deployed to various platforms:
- Vercel (recommended for Next.js)
- Netlify
- AWS Amplify
- Docker containers

## Configuration

### Environment Variables

```bash
# Backend API URL (default: http://localhost:8000)
NEXT_PUBLIC_API_URL=

# Additional configuration options
NEXT_PUBLIC_APP_ENV=production
```

### Customization

The application supports theming through Tailwind CSS configuration. Modify `tailwind.config.ts` to customize colors, spacing, and other design tokens.

## Troubleshooting

### Common Issues

1. **API connection errors**: Verify backend is running on correct port
2. **WebSocket connection fails**: Check CORS settings in backend
3. **Build failures**: Clear `.next` directory and node_modules, then reinstall

### Browser Support

- Chrome/Chromium 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## How It Works
