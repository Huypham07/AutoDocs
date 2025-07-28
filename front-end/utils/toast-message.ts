import { toast } from "sonner";

export const toastMessage = (message: string, type: "error" | "success" | "info") => {
  toast(message, {
    duration: 3000,
    className: type === "error" ? "text-red-700" : type === "success" ? "text-green-700" : "text-blue-700",
    actionButtonStyle: { backgroundColor: "ButtonShadow", color: "black" },
    position: "top-center",
    style: {
      backgroundColor: "white",
      outline: "1px solid #ccc",
    },
  });
};
