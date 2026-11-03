//! Just enough of the VP9 bitstream to tell which frames a packet holds are
//! never shown. FFmpeg's V4L2 wrapper hands the Frame's hardware decoder a
//! superframe whole, in one buffer, and the decoder gives back a picture
//! for each frame in it, hidden ones too, all with the packet's pts;
//! [`HiddenFrames`] picks those out so `Decoder::receive` can drop them.

use fp_ffmpeg_sys as ff;
use std::collections::VecDeque;

/// The frames of a packet: a superframe's, by the index at its end (VP9
/// bitstream specification, annex B), or the packet itself.
fn frames(data: &[u8]) -> Vec<&[u8]> {
    let Some(&last) = data.last() else {
        return Vec::new();
    };
    if last & 0xe0 == 0xc0 {
        let count = (last & 7) as usize + 1;
        let size_bytes = ((last >> 3) & 3) as usize + 1;
        let index = 2 + count * size_bytes;
        if let Some(start) = data.len().checked_sub(index)
            && data[start] == last
        {
            let mut rest = &data[..start];
            let mut out = Vec::with_capacity(count);
            for size in data[start + 1..data.len() - 1].chunks(size_bytes) {
                let size = size.iter().rev().fold(0, |n, &b| n << 8 | b as usize);
                let Some((frame, tail)) = rest.split_at_checked(size) else {
                    break;
                };
                out.push(frame);
                rest = tail;
            }
            return out;
        }
    }
    vec![data]
}

/// Whether a frame's uncompressed header says it is decoded but not shown:
/// `show_frame` 0, as alternate reference frames are. A frame showing an
/// earlier one (`show_existing_frame`) is shown.
fn hidden(frame: &[u8]) -> bool {
    let Some(&b) = frame.first() else {
        return false;
    };
    // frame_marker (2 bits), profile (low bit, then high bit), a reserved
    // bit in profile 3, show_existing_frame, frame_type, show_frame.
    if b >> 6 != 2 {
        return false;
    }
    let profile = (b >> 5 & 1) | (b >> 3 & 2);
    let show_existing = if profile == 3 { b >> 2 & 1 } else { b >> 3 & 1 };
    let show_frame = if profile == 3 { b & 1 } else { b >> 1 & 1 };
    show_existing == 0 && show_frame == 0
}

/// Frames in a packet that are decoded but never shown.
pub(crate) fn hidden_frames(data: &[u8]) -> u32 {
    frames(data).into_iter().filter(|f| hidden(f)).count() as u32
}

/// Hidden frames still to come out of a decoder that hands them back, by
/// the pts of the packet that held them. Each comes out right before its
/// packet's shown frame, with the same pts.
#[derive(Debug, Default)]
pub(crate) struct HiddenFrames {
    pending: VecDeque<(i64, u32)>,
    /// The pts of the frame just dropped, which the next one shares.
    dropped: Option<i64>,
}

/// Packets with hidden frames remembered at most: a decoder holds far fewer
/// in flight.
const PENDING_KEPT: usize = 64;

impl HiddenFrames {
    /// Notes the hidden frames of a packet the decoder took.
    pub(crate) fn sent(&mut self, pts: i64, data: &[u8]) {
        let n = hidden_frames(data);
        if n > 0 && pts != ff::AV_NOPTS_VALUE {
            self.pending.push_back((pts, n));
            if self.pending.len() > PENDING_KEPT {
                self.pending.pop_front();
            }
        }
    }

    /// Whether the decoded frame at `pts` is a hidden one, to drop. `None`
    /// when the frame after a dropped one has another pts: the decoder
    /// doesn't hand hidden frames back, and the one dropped was shown.
    pub(crate) fn drops(&mut self, pts: i64) -> Option<bool> {
        if self.dropped.take().is_some_and(|d| d != pts) {
            return None;
        }
        // Output follows pts: packets before this one are done.
        while self.pending.front().is_some_and(|&(p, _)| p < pts) {
            self.pending.pop_front();
        }
        match self.pending.front_mut() {
            Some((p, n)) if *p == pts => {
                *n -= 1;
                if *n == 0 {
                    self.pending.pop_front();
                }
                self.dropped = Some(pts);
                Some(true)
            }
            _ => Some(false),
        }
    }

    /// Forgets everything sent (a seek).
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.dropped = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame header byte: marker, profile, then show_existing_frame,
    /// frame_type (inter) and show_frame.
    fn frame(profile: u8, show_existing: bool, show_frame: bool) -> Vec<u8> {
        let p = (profile & 1) << 5 | (profile >> 1 & 1) << 4;
        let bits = if profile == 3 {
            (show_existing as u8) << 2 | 1 << 1 | show_frame as u8
        } else {
            (show_existing as u8) << 3 | 1 << 2 | (show_frame as u8) << 1
        };
        vec![0x80 | p | bits, 0x55, 0xaa]
    }

    /// A superframe of `frames` with `size_bytes`-byte sizes in its index.
    fn superframe(frames: &[Vec<u8>], size_bytes: usize) -> Vec<u8> {
        let marker = 0xc0 | ((size_bytes - 1) as u8) << 3 | (frames.len() - 1) as u8;
        let mut out: Vec<u8> = frames.concat();
        out.push(marker);
        for f in frames {
            out.extend_from_slice(&f.len().to_le_bytes()[..size_bytes]);
        }
        out.push(marker);
        out
    }

    #[test]
    fn finds_hidden_frames() {
        let (shown, alt_ref, existing) = (
            frame(0, false, true),
            frame(0, false, false),
            frame(0, true, false),
        );
        // An alternate reference packed ahead of a shown frame, as libvpx
        // and YouTube's files have it, with 1- and 2-byte sizes.
        assert_eq!(
            hidden_frames(&superframe(&[alt_ref.clone(), shown.clone()], 1)),
            1
        );
        assert_eq!(
            hidden_frames(&superframe(
                &[alt_ref.clone(), alt_ref.clone(), shown.clone()],
                2
            )),
            2
        );
        assert_eq!(hidden_frames(&alt_ref), 1);
        for shown in [
            shown.clone(),
            existing,
            frame(3, false, true),
            frame(3, true, false),
        ] {
            assert_eq!(hidden_frames(&shown), 0, "{shown:02x?}");
        }
        assert_eq!(hidden_frames(&frame(3, false, false)), 1);
        // A last byte that looks like an index but isn't one.
        let mut plain = shown.clone();
        plain.push(0xc1);
        assert_eq!(hidden_frames(&plain), 0);
        assert_eq!(hidden_frames(&[]), 0);
    }

    #[test]
    fn drops_each_hidden_frame_before_its_shown_one() {
        let packet = superframe(&[frame(0, false, false), frame(0, false, true)], 1);
        let mut h = HiddenFrames::default();
        h.sent(0, &frame(0, false, true));
        h.sent(17, &packet);
        h.sent(33, &frame(0, false, true));
        let out: Vec<_> = [0, 17, 17, 33].iter().map(|&pts| h.drops(pts)).collect();
        assert_eq!(out, [Some(false), Some(true), Some(false), Some(false)]);

        // A decoder that keeps hidden frames to itself: the shown frame
        // dropped is noticed at the next one.
        let mut h = HiddenFrames::default();
        h.sent(17, &packet);
        assert_eq!(h.drops(17), Some(true));
        assert_eq!(h.drops(33), None);

        // Nothing carries over a seek.
        let mut h = HiddenFrames::default();
        h.sent(17, &packet);
        h.clear();
        assert_eq!(h.drops(17), Some(false));
    }
}
