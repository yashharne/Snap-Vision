package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

func mergeChunks (chunkPaths []string, outputVideo string) error {
	if len(chunkPaths) == 0 {
		return fmt.Errorf("no chunks to merge")
	}

	listFile, err := os.Create("list.txt")
	if err != nil {
			return fmt.Errorf("create list file failed: %w", err)
	}
	defer os.Remove("list.txt")
	defer listFile.Close()

	for _, chunkPath := range chunkPaths {
			_, err := listFile.WriteString(fmt.Sprintf("file '%s'\n", chunkPath))
			if err != nil {
					return fmt.Errorf("write to list file failed: %w", err)
			}
	}

	cmd := exec.Command("ffmpeg", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", outputVideo)

	if err := cmd.Run(); err != nil {
			return fmt.Errorf("ffmpeg merge failed: %w", err)
	}

	return nil
}
func main() {
    chunkDir := "chunks"
	files, err := filepath.Glob(filepath.Join(chunkDir, "chunk*.mp4"))
	if err != nil {
			log.Fatalf("glob failed: %v", err)
	}

	sort.Slice(files, func(i, j int) bool {
			iName := filepath.Base(files[i])
			jName := filepath.Base(files[j])

			iNumStr := strings.TrimPrefix(iName, "chunk")
			jNumStr := strings.TrimPrefix(jName, "chunk")

			iNum := 0
			jNum := 0

			fmt.Sscanf(iNumStr, "%d", &iNum)
			fmt.Sscanf(jNumStr, "%d", &jNum)
			return iNum < jNum
	})

	if err := mergeChunks(files, "snapped_video.mp4"); err != nil {
			log.Fatalf("merge chunks failed: %v", err)
	}

	fmt.Println("Merged video saved as snapped_video.mp4")
}