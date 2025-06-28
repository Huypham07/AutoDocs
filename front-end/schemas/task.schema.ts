export interface TaskBody {
  repo_url: string;
  access_token?: string;
}

export interface TaskResponse {
  owner: string;
  repo_name: string;
}
