<script lang="ts">
  import { SplitPane } from "@rich_harris/svelte-split-pane";
  import { ExternalLinkIcon, PlayIcon, TerminalIcon } from "lucide-svelte";

  import ReplEditor from "./ReplEditor.svelte";
  import { encodeContent } from "./encode";

  let {
    initialText,
  }: {
    initialText: string;
  } = $props();

  let editor: ReplEditor;
  let currentText = $state(initialText);
  let replLink = $derived(`/repl?content=` + encodeContent(currentText));
</script>

<div class="h-full flex flex-col border border-gray-200 rounded-lg">
  <div
    class="shrink-0 border-b border-gray-200 flex items-center px-2 py-1.5 gap-2"
  >
    <button
      class="bg-green-100 hover:bg-green-200 px-2 py-0.5 rounded flex text-sm items-center gap-1.5"
    >
      <PlayIcon size={16} />
      Run
    </button>
    <a
      href={replLink}
      class="hover:bg-gray-100 px-2 py-0.5 rounded flex text-sm items-center gap-1.5 text-gray-600"
    >
      <ExternalLinkIcon size={16} />
      Open REPL
    </a>

    <div class="ml-auto"></div>
    <select class="hover:bg-gray-100 rounded text-sm pt-[3px] py-0.5">
      <option>WebGPU</option>
      <option>Wasm</option>
      <option>CPU (slow)</option>
    </select>
  </div>
  <div class="flex-1 min-h-0">
    <SplitPane
      type="vertical"
      pos="-40px"
      min="40px"
      max="-40px"
      --color="var(--color-gray-200)"
    >
      {#snippet a()}
        <ReplEditor
          bind:this={editor}
          {initialText}
          editorOptions={{
            lineNumbersMinChars: 4,
            padding: {
              top: 8,
              bottom: 8,
            },
            minimap: { enabled: false },
            scrollbar: { alwaysConsumeMouseWheel: false, useShadows: false },
            scrollBeyondLastLine: false,
          }}
          onchange={() => {
            currentText = editor.getText();
          }}
        />
      {/snippet}
      {#snippet b()}
        <div class="flex items-center justify-center select-none">
          <TerminalIcon size={20} class="text-gray-300 mr-2" />
          <p class="text-sm text-gray-500">Run code to see output here.</p>
        </div>
      {/snippet}
    </SplitPane>
  </div>
</div>
