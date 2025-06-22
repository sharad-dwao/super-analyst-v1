import Link from "next/link";
import React, { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const NonMemoizedMarkdown = ({ children }: { children: string }) => {
  const components: Partial<Components> = {
    // @ts-expect-error
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        // @ts-expect-error
        <pre
          {...props}
          className={`${className} text-sm w-full max-w-full overflow-x-auto bg-muted p-3 rounded-lg mt-2 border border-border`}
        >
          <code className={match[1]}>{children}</code>
        </pre>
      ) : (
        <code
          className={`${className} text-sm bg-muted py-0.5 px-1 rounded-md border border-border`}
          {...props}
        >
          {children}
        </code>
      );
    },
    ol: ({ node, children, ...props }) => {
      return (
        <ol className="list-decimal list-outside ml-4 space-y-1" {...props}>
          {children}
        </ol>
      );
    },
    li: ({ node, children, ...props }) => {
      return (
        <li className="py-0.5" {...props}>
          {children}
        </li>
      );
    },
    ul: ({ node, children, ...props }) => {
      return (
        <ul className="list-disc list-outside ml-4 space-y-1" {...props}>
          {children}
        </ul>
      );
    },
    strong: ({ node, children, ...props }) => {
      return (
        <span className="font-semibold text-foreground" {...props}>
          {children}
        </span>
      );
    },
    a: ({ node, children, ...props }) => {
      return (
        // @ts-expect-error
        <Link
          className="text-primary hover:text-primary/80 hover:underline transition-colors"
          target="_blank"
          rel="noreferrer"
          {...props}
        >
          {children}
        </Link>
      );
    },
    h1: ({ node, children, ...props }) => {
      return (
        <h1 className="text-3xl font-semibold mt-6 mb-3 text-foreground" {...props}>
          {children}
        </h1>
      );
    },
    h2: ({ node, children, ...props }) => {
      return (
        <h2 className="text-2xl font-semibold mt-6 mb-3 text-foreground" {...props}>
          {children}
        </h2>
      );
    },
    h3: ({ node, children, ...props }) => {
      return (
        <h3 className="text-xl font-semibold mt-5 mb-2 text-foreground" {...props}>
          {children}
        </h3>
      );
    },
    h4: ({ node, children, ...props }) => {
      return (
        <h4 className="text-lg font-semibold mt-4 mb-2 text-foreground" {...props}>
          {children}
        </h4>
      );
    },
    h5: ({ node, children, ...props }) => {
      return (
        <h5 className="text-base font-semibold mt-4 mb-2 text-foreground" {...props}>
          {children}
        </h5>
      );
    },
    h6: ({ node, children, ...props }) => {
      return (
        <h6 className="text-sm font-semibold mt-3 mb-2 text-foreground" {...props}>
          {children}
        </h6>
      );
    },
    p: ({ node, children, ...props }) => {
      return (
        <p className="mb-3 leading-relaxed text-foreground" {...props}>
          {children}
        </p>
      );
    },
    blockquote: ({ node, children, ...props }) => {
      return (
        <blockquote className="border-l-4 border-border pl-4 my-4 italic text-muted-foreground bg-muted/30 py-2 rounded-r-md" {...props}>
          {children}
        </blockquote>
      );
    },
    table: ({ node, children, ...props }) => {
      return (
        <div className="overflow-x-auto my-4">
          <table className="min-w-full border border-border rounded-lg" {...props}>
            {children}
          </table>
        </div>
      );
    },
    thead: ({ node, children, ...props }) => {
      return (
        <thead className="bg-muted" {...props}>
          {children}
        </thead>
      );
    },
    th: ({ node, children, ...props }) => {
      return (
        <th className="border border-border px-3 py-2 text-left font-semibold" {...props}>
          {children}
        </th>
      );
    },
    td: ({ node, children, ...props }) => {
      return (
        <td className="border border-border px-3 py-2" {...props}>
          {children}
        </td>
      );
    },
  };

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  );
};

export const Markdown = memo(
  NonMemoizedMarkdown,
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);