"use client";

import type { ChatRequestOptions, CreateMessage, Message } from "ai";
import { motion, AnimatePresence } from "framer-motion";
import type React from "react";
import {
  useRef,
  useEffect,
  useCallback,
  type Dispatch,
  type SetStateAction,
} from "react";
import { toast } from "sonner";
import { useLocalStorage, useWindowSize } from "usehooks-ts";

import { cn, sanitizeUIMessages } from "@/lib/utils";

import { ArrowUpIcon, StopIcon } from "./icons";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

const suggestedActions = [
  {
    title: "What were the top 10 most",
    label: "visited pages yesterday?",
    action: "What were the top 10 most visited pages yesterday?",
  },
  {
    title: "Which pages had the highest",
    label: "bounce rate this month?",
    action: "Which pages had the highest bounce rate this month?",
  },
  {
    title: "Show me traffic trends",
    label: "by marketing channel",
    action: "Show me traffic trends by marketing channel for the last month",
  },
  {
    title: "Compare desktop vs mobile",
    label: "conversion rates",
    action: "Compare desktop vs mobile conversion rates for the last 30 days",
  },
];

export function MultimodalInput({
  chatId,
  input,
  setInput,
  isLoading,
  stop,
  messages,
  setMessages,
  append,
  handleSubmit,
  className,
}: {
  chatId: string;
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  stop: () => void;
  messages: Array<Message>;
  setMessages: Dispatch<SetStateAction<Array<Message>>>;
  append: (
    message: Message | CreateMessage,
    chatRequestOptions?: ChatRequestOptions,
  ) => Promise<string | null | undefined>;
  handleSubmit: (
    event?: {
      preventDefault?: () => void;
    },
    chatRequestOptions?: ChatRequestOptions,
  ) => void;
  className?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { width } = useWindowSize();

  useEffect(() => {
    if (textareaRef.current) {
      adjustHeight();
    }
  }, []);

  const adjustHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight + 2}px`;
    }
  };

  const [localStorageInput, setLocalStorageInput] = useLocalStorage(
    "input",
    "",
  );

  useEffect(() => {
    if (textareaRef.current) {
      const domValue = textareaRef.current.value;
      // Prefer DOM value over localStorage to handle hydration
      const finalValue = domValue || localStorageInput || "";
      setInput(finalValue);
      adjustHeight();
    }
    // Only run once after hydration
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setLocalStorageInput(input);
  }, [input, setLocalStorageInput]);

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
    adjustHeight();
  };

  const submitForm = useCallback(() => {
    handleSubmit(undefined, {});
    setLocalStorageInput("");

    if (width && width > 768) {
      textareaRef.current?.focus();
    }
  }, [handleSubmit, setLocalStorageInput, width]);

  return (
    <div className="relative w-full flex flex-col gap-4">
      <AnimatePresence>
        {messages.length === 0 && (
          <motion.div 
            className="grid sm:grid-cols-2 gap-2 w-full"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {suggestedActions.map((suggestedAction, index) => (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ delay: 0.05 * index, duration: 0.3 }}
                key={`suggested-action-${suggestedAction.title}-${index}`}
                className={index > 1 ? "hidden sm:block" : "block"}
              >
                <Button
                  variant="ghost"
                  onClick={async () => {
                    append({
                      role: "user",
                      content: suggestedAction.action,
                    });
                  }}
                  className="text-left border rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start hover:bg-accent/50 transition-colors"
                  disabled={isLoading}
                >
                  <span className="font-medium">{suggestedAction.title}</span>
                  <span className="text-muted-foreground">
                    {suggestedAction.label}
                  </span>
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        className="relative"
        animate={isLoading ? { scale: [1, 1.01, 1] } : { scale: 1 }}
        transition={{ duration: 2, repeat: isLoading ? Infinity : 0 }}
      >
        <Textarea
          ref={textareaRef}
          placeholder={isLoading ? "Processing your request..." : "Ask about your website analytics data..."}
          value={input}
          onChange={handleInput}
          disabled={isLoading}
          className={cn(
            "min-h-[60px] max-h-[200px] overflow-hidden resize-none rounded-xl !text-base bg-muted text-foreground placeholder:text-muted-foreground border-border focus-visible:ring-1 focus-visible:ring-ring transition-all",
            isLoading && "opacity-75 cursor-not-allowed",
            className,
          )}
          rows={3}
          autoFocus
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();

              if (isLoading) {
                toast.error("Please wait for the model to finish its response!");
              } else {
                submitForm();
              }
            }
          }}
        />

        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="stop-button"
              initial={{ scale: 0, opacity: 0, rotate: -90 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0, opacity: 0, rotate: 90 }}
              transition={{ duration: 0.2 }}
            >
              <Button
                className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border border-destructive/20 bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors shadow-lg"
                onClick={(event) => {
                  event.preventDefault();
                  stop();
                  setMessages((messages) => sanitizeUIMessages(messages));
                }}
              >
                <StopIcon size={14} />
              </Button>
            </motion.div>
          ) : (
            <motion.div
              key="send-button"
              initial={{ scale: 0, opacity: 0, rotate: -90 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0, opacity: 0, rotate: 90 }}
              transition={{ duration: 0.2 }}
            >
              <Button
                className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border border-primary/20 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 shadow-lg"
                onClick={(event) => {
                  event.preventDefault();
                  submitForm();
                }}
                disabled={input.length === 0}
              >
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  transition={{ duration: 0.1 }}
                >
                  <ArrowUpIcon size={14} />
                </motion.div>
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}