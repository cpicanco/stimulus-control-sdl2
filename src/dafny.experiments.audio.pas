{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit dafny.experiments.audio;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils;

type
  TWord = record
    IPA : string;
    Caption : string;
  end;

  TWords = array of TWord;

var
  Words : TWords;

const
  bog = 'bɔgui';
  bot = 'bɔti';
  fek = 'fɛki';
  fep = 'fɛpi';
  jek = 'ʒɛki';
  jep = 'ʒɛpi';
  kag = 'kagui';
  kib = 'kibi';
  kog = 'kɔgui';
  mag = 'magui';
  mat = 'mati';
  med = 'mɛdi';
  mib = 'mibi';
  min = 'mini';
  mob = 'mɔbi';
  mog = 'mɔgui';
  mot = 'mɔti';
  mup = 'mupi';
  nag = 'nagui';
  nek = 'nɛki';
  nep = 'nɛpi';
  nib = 'nibi';
  nog = 'nɔgui';
  pag = 'pagui';
  pib = 'pibi';
  pog = 'pɔgui';
  rek = 'ʁɛki';
  rep = 'ʁɛpi';
  rog = 'ʁɔgui';
  rot = 'ʁɔti';
  sag = 'sagui';
  sat = 'sati';
  sed = 'sɛdi';
  sib = 'sibi';
  sin = 'sini';
  sob = 'sɔbi';
  sog = 'sɔgui';
  sot = 'sɔti';
  sup = 'supi';
  tag = 'tagui';
  tek = 'tɛki';
  tep = 'tɛpi';
  tib = 'tibi';
  tog = 'tɔgui';



  HR_bog = 'bog';
  HR_bot = 'bot';
  HR_fek = 'fek';
  HR_fep = 'fep';
  HR_jek = 'jek';
  HR_jep = 'jep';
  HR_kag = 'kag';
  HR_kib = 'kib';
  HR_kog = 'kog';
  HR_mag = 'mag';
  HR_mat = 'mat';
  HR_med = 'med';
  HR_mib = 'mib';
  HR_min = 'min';
  HR_mob = 'mob';
  HR_mog = 'mog';
  HR_mot = 'mot';
  HR_mup = 'mup';
  HR_nag = 'nag';
  HR_nek = 'nek';
  HR_nep = 'nep';
  HR_nib = 'nib';
  HR_nog = 'nog';
  HR_pag = 'pag';
  HR_pib = 'pib';
  HR_pog = 'pog';
  HR_rek = 'rek';
  HR_rep = 'rep';
  HR_rog = 'rog';
  HR_rot = 'rot';
  HR_sag = 'sag';
  HR_sat = 'sat';
  HR_sed = 'sed';
  HR_sib = 'sib';
  HR_sin = 'sin';
  HR_sob = 'sob';
  HR_sog = 'sog';
  HR_sot = 'sot';
  HR_sup = 'sup';
  HR_tag = 'tag';
  HR_tek = 'tek';
  HR_tep = 'tep';
  HR_tib = 'tib';
  HR_tog = 'tog';

var

  IPAWords : array [0..43] of string = (
    bog,bot,fek,fep,jek,jep,
    kag,kib,kog,mag,mat,med,
    mib,min,mob,mog,mot,mup,
    nag,nek,nep,nib,nog,pag,
    pib,pog,rek,rep,rog,rot,
    sag,sat,sed,sib,sin,sob,
    sog,sot,sup,tag,tek,tep,
    tib,tog);

  HRWords : array [0..43] of string = (
    HR_bog, HR_bot, HR_fek, HR_fep, HR_jek, HR_jep,
    HR_kag, HR_kib, HR_kog, HR_mag, HR_mat, HR_med,
    HR_mib, HR_min, HR_mob, HR_mog, HR_mot, HR_mup,
    HR_nag, HR_nek, HR_nep, HR_nib, HR_nog, HR_pag,
    HR_pib, HR_pog, HR_rek, HR_rep, HR_rog, HR_rot,
    HR_sag, HR_sat, HR_sed, HR_sib, HR_sin, HR_sob,
    HR_sog, HR_sot, HR_sup, HR_tag, HR_tek, HR_tep,
    HR_tib, HR_tog);

  BlocksOnset: array [0..5] of array [0..1] of string = (
    ('mat', 'sat'),
    ('mib', 'sib'),
    ('mob', 'sob'),
    ('mup', 'sup'),
    ('min', 'sin'),
    ('med', 'sed')
  );

  BlocksRime: array [0..5] of array [0..1] of string = (
    ('kog', 'kib'),
    ('sog', 'sib'),
    ('nog', 'nib'),
    ('tog', 'tib'),
    ('mog', 'mib'),
    ('pog', 'pib')
  );

  BlocksCoda: array [0..8] of array [0..1] of string = (
    ('mot', 'mog'),
    ('sot', 'sog'),
    ('bot', 'bog'),
    ('rot', 'rog'),
    ('tep', 'tek'),
    ('rep', 'rek'),
    ('nep', 'nek'),
    ('fep', 'fek'),
    ('jep', 'jek')
  );

  BlocksVowel: array [0..5] of array [0..1] of string = (
    ('kog', 'kag'),
    ('sog', 'sag'),
    ('nog', 'nag'),
    ('tog', 'tag'),
    ('mog', 'mag'),
    ('pog', 'pag')
  );

procedure Synthetize;
procedure InitializeWords;
function Initialized : Boolean;

implementation

uses media.audio;

function SSML(IPA: string; Caption: string) : string;
begin
  Result :=
  '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">' +
    '<voice name="Microsoft Maria Desktop - Portuguese(Brazil)">' +
      '<prosody rate="10%">' +
          '<phoneme alphabet="ipa" ph="'+IPA+'">'+Caption+'</phoneme>' +
      '</prosody>'+
    '</voice>'+
  '</speak>';
end;

function Initialized : Boolean;
begin
  Result := Length(Words) = Length(IPAWords);
end;

procedure InitializeWords;
var
  i: Integer;
begin
  SetLength(Words, Length(IPAWords));
  for i := Low(IPAWords) to High(IPAWords) do begin
    Words[i].IPA := IPAWords[i];
    Words[i].Caption := HRWords[i];
  end;
end;

procedure Synthetize;
var
  LWord : TWord;

  function GetPath(AFilename : string):string;
  var
    LPath : string;
  begin
    LPath := ConcatPaths(['wav', 'microsoft-daap', AFilename+'.wav']);
    Result := GetMediaPath(LPath);
  end;
begin
  for LWord in Words do
    SpeakToFile(LWord.IPA, GetPath(LWord.Caption));
end;

end.

