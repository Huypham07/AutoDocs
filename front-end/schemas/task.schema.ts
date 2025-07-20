export interface TaskBody {
  repo_url: string;
  access_token?: string;
}

export interface DocumantationResponse {
  id: string;
  title: string;
  description: string;
  owner: string;
  repo_name: string;
  repo_url: string;
  created_at: string;
  updated_at: string;
  status: string;
  root_sections: Section[];
}

export interface Section {
  section_id: string;
  section_title: string;
  pages: Page[];
  subsections: Section[];
}

export interface Page {
  page_id: string;
  page_title: string;
  content: string;
  file_paths: string[];
}
