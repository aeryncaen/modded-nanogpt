// retokenize replaces removed BPE tokens (>16 bytes) in pre-tokenized .bin files.
//
// The .bin format is: 256 int32 header + packed uint16 tokens.
// For each of the 72 removed token IDs, we splice in the replacement sequence
// (deterministic BPE decomposition into shorter tokens).
//
// Usage: go run retokenize.go <input.bin> <output.bin>
//    or: go run retokenize.go <dir>   (processes all .bin files in-place)

package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const headerSize = 256 * 4 // 256 int32s = 1024 bytes

// Replacement map: old token ID -> slice of new token IDs
// Generated from tiktoken: each removed token's bytes re-encoded with trimmed tokenizer
var replacements = map[uint16][]uint16{
	3880:  {1783, 1783},
	8864:  {4181, 4181},
	10052: {4770, 4770},
	10097: {1783, 1783, 1783, 1783},
	10221: {4841, 4841},
	14827: {9364, 9364},
	14950: {8184, 8184},
	15171: {2424, 7992},
	16529: {220, 1783, 1783, 1783, 1783},
	17174: {8412, 8412},
	19351: {1783, 650},
	20368: {220, 1783, 1783},
	20727: {555, 18789},
	22369: {1783, 982},
	23090: {9364, 9364, 9364, 9364},
	23193: {4181, 4181, 4181, 4181},
	23926: {4770, 4770, 4770, 4770},
	27006: {15243, 15243},
	27193: {4841, 4841, 4841, 4841},
	27473: {5735, 20860},
	27754: {4181, 2109},
	28542: {16068, 16068},
	28719: {18717, 1286},
	29113: {14468, 14468},
	29146: {15864, 15864},
	29760: {15149, 28018},
	29789: {16782, 5646},
	30210: {29372, 18143},
	30213: {30212, 10049},
	30542: {8184, 8184, 8184, 8184},
	30899: {21018, 30898},
	30906: {30905, 21018, 30898},
	30982: {18717, 378},
	31576: {22615, 31573},
	32799: {2095, 1634},
	32941: {4841, 2602},
	34400: {220, 1783},
	35496: {9364, 9364, 9364, 9364, 9364, 9364, 9364, 9364},
	36174: {36173, 35992},
	36573: {3753, 19541},
	36658: {796, 4770},
	37389: {26825, 12100},
	38093: {796, 4770, 4770, 4770, 4770},
	39172: {17811, 17811},
	39177: {7449, 39142},
	39753: {39752, 10493},
	39755: {39714, 39655},
	39756: {24807, 31208},
	39757: {17620, 29841},
	40242: {39693, 40241},
	40586: {6142, 1023},
	40800: {11784, 453},
	40887: {33131, 36387},
	41380: {21353, 41215},
	41436: {220, 1783, 650},
	41906: {220, 8412, 8412},
	42045: {11273, 16607},
	43453: {15831, 42202},
	43649: {37665, 41726},
	43801: {1783, 1783, 1783, 982},
	44436: {17038, 1056},
	44713: {220, 4181},
	45545: {45544, 42983},
	45706: {22686, 22686},
	46111: {796, 4770, 4770},
	46674: {3753, 27781},
	47232: {1783, 1783, 1783},
	47757: {13352, 34718},
	48667: {14318, 20860},
	49129: {4181, 492},
	49527: {20503, 20503},
	49704: {27246, 27246},
}

func processFile(path string, outPath string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}

	if len(data) < headerSize {
		return fmt.Errorf("%s: file too small for header", path)
	}

	// Parse header
	header := data[:headerSize]
	magic := binary.LittleEndian.Uint32(header[0:4])
	version := binary.LittleEndian.Uint32(header[4:8])
	numTokens := binary.LittleEndian.Uint32(header[8:12])

	if magic != 20240520 {
		return fmt.Errorf("%s: bad magic %d", path, magic)
	}
	if version != 1 {
		return fmt.Errorf("%s: unsupported version %d", path, version)
	}

	tokenData := data[headerSize:]
	expectedBytes := int(numTokens) * 2
	if len(tokenData) < expectedBytes {
		return fmt.Errorf("%s: expected %d token bytes, got %d", path, expectedBytes, len(tokenData))
	}

	// Read tokens
	tokens := make([]uint16, numTokens)
	for i := range tokens {
		tokens[i] = binary.LittleEndian.Uint16(tokenData[i*2 : i*2+2])
	}

	// Count replacements needed
	replCount := 0
	extraTokens := 0
	for _, t := range tokens {
		if rep, ok := replacements[t]; ok {
			replCount++
			extraTokens += len(rep) - 1 // net change in token count
		}
	}

	if replCount == 0 {
		fmt.Printf("  %s: no replacements needed, copying\n", filepath.Base(path))
		if outPath != path {
			return os.WriteFile(outPath, data, 0644)
		}
		return nil
	}

	// Build new token sequence
	newTokens := make([]uint16, 0, int(numTokens)+extraTokens)
	for _, t := range tokens {
		if rep, ok := replacements[t]; ok {
			newTokens = append(newTokens, rep...)
		} else {
			newTokens = append(newTokens, t)
		}
	}

	// Update header with new token count
	newHeader := make([]byte, headerSize)
	copy(newHeader, header)
	binary.LittleEndian.PutUint32(newHeader[8:12], uint32(len(newTokens)))

	// Write output
	out := make([]byte, headerSize+len(newTokens)*2)
	copy(out, newHeader)
	for i, t := range newTokens {
		binary.LittleEndian.PutUint16(out[headerSize+i*2:], t)
	}

	if err := os.WriteFile(outPath, out, 0644); err != nil {
		return fmt.Errorf("write %s: %w", outPath, err)
	}

	fmt.Printf("  %s: %d replacements, %d -> %d tokens\n",
		filepath.Base(path), replCount, numTokens, len(newTokens))
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <file.bin> [output.bin]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "       %s <directory>  (in-place)\n", os.Args[0])
		os.Exit(1)
	}

	target := os.Args[1]
	info, err := os.Stat(target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if info.IsDir() {
		// Process all .bin files in directory
		matches, _ := filepath.Glob(filepath.Join(target, "*.bin"))
		if len(matches) == 0 {
			fmt.Fprintf(os.Stderr, "No .bin files found in %s\n", target)
			os.Exit(1)
		}
		fmt.Printf("Processing %d .bin files in %s\n", len(matches), target)
		for _, f := range matches {
			if err := processFile(f, f); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
		}
		fmt.Println("Done.")
	} else {
		// Single file mode
		outPath := target
		if len(os.Args) >= 3 {
			outPath = os.Args[2]
		}
		if !strings.HasSuffix(target, ".bin") {
			fmt.Fprintf(os.Stderr, "Warning: %s doesn't end in .bin\n", target)
		}
		if err := processFile(target, outPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}
