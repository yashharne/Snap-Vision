import { createClient } from "@supabase/supabase-js";

const projecturl = import.meta.env.VITE_PROJECT_URL;
const anon_key = import.meta.env.VITE_ANON_KEY;
const service_role_key = import.meta.env.VITE_SERVICE_ROLE_KEY;

// Create Supabase client
export const supabase = createClient(projecturl, service_role_key);
