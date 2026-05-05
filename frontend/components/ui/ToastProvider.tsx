"use client";

import * as Toast from "@radix-ui/react-toast";
import React, { createContext, useCallback, useContext, useState } from "react";
import { cn } from "@/lib/utils";

interface ToastMessage {
  id: string;
  title: string;
  description?: string;
  variant?: "default" | "destructive";
}

interface ToastContextValue {
  toast: (msg: Omit<ToastMessage, "id">) => void;
}

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast(): ToastContextValue {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<ToastMessage[]>([]);

  const toast = useCallback((msg: Omit<ToastMessage, "id">) => {
    const id = Math.random().toString(36).slice(2);
    setMessages((prev) => [...prev, { ...msg, id }]);
    setTimeout(() => {
      setMessages((prev) => prev.filter((m) => m.id !== id));
    }, 5000);
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      <Toast.Provider swipeDirection="right" duration={5000}>
        {children}
        {messages.map((m) => (
          <Toast.Root
            key={m.id}
            className={cn(
              "flex flex-col gap-1 rounded-lg border p-4 shadow-xl",
              "data-[state=open]:animate-in data-[state=closed]:animate-out",
              "data-[swipe=end]:animate-out data-[state=closed]:fade-out-80",
              "data-[state=open]:slide-in-from-bottom-4",
              m.variant === "destructive"
                ? "border-red-800 bg-red-950 text-red-100"
                : "border-[#2a2a2a] bg-[#141414] text-white",
            )}
          >
            <Toast.Title className="text-sm font-semibold">{m.title}</Toast.Title>
            {m.description && (
              <Toast.Description className="text-xs text-[#888]">{m.description}</Toast.Description>
            )}
          </Toast.Root>
        ))}
        <Toast.Viewport className="fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2" />
      </Toast.Provider>
    </ToastContext.Provider>
  );
}
