import React, { useState, useEffect } from "react";
import { View, Text, FlatList, TouchableOpacity } from "react-native";
import { sendFeedback } from "../har/flClient";
import { type Window as MLWindow } from "../har/features";

interface FeedbackSectionProps {
  currentWindowData: MLWindow | null;
}

export default function FeedbackSection({ currentWindowData }: FeedbackSectionProps) {
  const [activeLabel, setActiveLabel] = useState<string | null>(null);
  const [lastSent, setLastSent] = useState<number>(0);

  const activityLabels = [
    "Duduk Aktif",
    "Berbaring Aktif",
    "Berdiri Aktif",
    "Berdiri Tidak Aktif",
    "Duduk Tidak Aktif",
    "Berbaring Tidak Aktif",
  ];

  // streaming feedback otomatis setiap 2 detik
  useEffect(() => {
    if (!activeLabel || !currentWindowData) return;

    const now = Date.now();
    if (now - lastSent > 2000) { // setiap 2 detik
      sendFeedback(currentWindowData, activeLabel);
      setLastSent(now);
    }
  }, [currentWindowData, activeLabel, lastSent]);

  const handleSelect = (label: string) => {
    // toggle label aktif
    setActiveLabel(activeLabel === label ? null : label);
  };

  return (
    <View style={{ marginTop: 16 }}>
      <Text style={{ fontWeight: "bold", marginBottom: 8 }}>Mode Feedback Otomatis:</Text>

      <FlatList
        data={activityLabels}
        horizontal
        keyExtractor={(item) => item}
        renderItem={({ item }) => (
          <TouchableOpacity
            onPress={() => handleSelect(item)}
            style={{
              backgroundColor: activeLabel === item ? "#007AFF" : "#eee",
              paddingHorizontal: 12,
              paddingVertical: 8,
              borderRadius: 12,
              marginRight: 8,
            }}
          >
            <Text
              style={{
                color: activeLabel === item ? "white" : "black",
                fontWeight: "600",
              }}
            >
              {item}
            </Text>
          </TouchableOpacity>
        )}
      />

      {activeLabel ? (
        <Text style={{ marginTop: 10, color: "#007AFF" }}>
          🔄 Mengirim feedback otomatis: <Text style={{ fontWeight: "bold" }}>{activeLabel}</Text>
        </Text>
      ) : (
        <Text style={{ marginTop: 10, color: "gray" }}>Feedback nonaktif</Text>
      )}
    </View>
  );
}

















// import React, { useState } from "react";
// import { View, Text, Button, FlatList, TouchableOpacity, Alert } from "react-native";
// import { sendFeedback } from "../har/flClient";
// import { type Window as MLWindow } from "../har/features";  // ✅ tambahkan ini

// // ✅ ubah tipe props
// interface FeedbackSectionProps {
//   currentWindowData: MLWindow | null;
// }

// export default function FeedbackSection({ currentWindowData }: FeedbackSectionProps) {
//   const [selectedLabel, setSelectedLabel] = useState<string | null>(null);

//   const activityLabels = [
//     "Duduk Aktif",
//     "Berbaring Aktif",
//     "Berdiri Aktif",
//     "Berdiri Tidak Aktif",
//     "Duduk Tidak Aktif",
//     "Berbaring Tidak Aktif",
//   ];

//   const handleSend = () => {
//     if (!selectedLabel) {
//       Alert.alert("Pilih aktivitas", "Silakan pilih aktivitas yang benar terlebih dahulu.");
//       return;
//     }
//     if (!currentWindowData) {
//       Alert.alert("Belum ada data", "Tunggu window sensor terbentuk dulu.");
//       return;
//     }

//     sendFeedback(currentWindowData, selectedLabel);
//     Alert.alert("Feedback dikirim", `✅ ${selectedLabel}`);
//     setSelectedLabel(null);
//   };

//   return (
//     <View style={{ marginTop: 16 }}>
//       <Text style={{ fontWeight: "bold", marginBottom: 8 }}>Koreksi Aktivitas:</Text>

//       <FlatList
//         data={activityLabels}
//         horizontal
//         keyExtractor={(item) => item}
//         renderItem={({ item }) => (
//           <TouchableOpacity
//             onPress={() => setSelectedLabel(item)}
//             style={{
//               backgroundColor: selectedLabel === item ? "#007AFF" : "#eee",
//               paddingHorizontal: 12,
//               paddingVertical: 8,
//               borderRadius: 12,
//               marginRight: 8,
//             }}
//           >
//             <Text
//               style={{
//                 color: selectedLabel === item ? "white" : "black",
//                 fontWeight: "600",
//               }}
//             >
//               {item}
//             </Text>
//           </TouchableOpacity>
//         )}
//       />

//       <Button title="Kirim Koreksi" onPress={handleSend} />
//     </View>
//   );
// }
