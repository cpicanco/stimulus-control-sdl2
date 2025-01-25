{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.audio;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, SpeechLib_TLB;


implementation

uses
  //sdl.app.output
    media.audio
  , picanco.experiments.words.types
  , picanco.experiments.words;

type
  TPhonemeSet = (phIPA, phSYM);

function WordToSSML(const AWord : TWord;
  const APhonemeSet : TPhonemeSet = phIPA) : string;
begin
  Result :=
  '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">' +
    '<voice name="Microsoft Maria Desktop - Portuguese(Brazil)">' +
      '<prosody rate="10%">';
      case APhonemeSet of
        phIPA : Result := Result +
          '<phoneme alphabet="ipa" ph="'+AWord.ToIPA+'">'+AWord.Caption+'</phoneme>';
        phSYM : Result := Result +
          '<PRON SYM="'+AWord.ToSYM+'"/>'+AWord.Caption;
      end;
  Result := Result +
      '</prosody>'+
    '</voice>'+
  '</speak>';
end;

procedure SpeakWord(const AWord: TWord; const APhonemeSet: TPhonemeSet);
begin
  Speak(WordToSSML(AWord, APhonemeSet));
end;

procedure SpeakWordToFile(const AWord: TWord;
  APhonemeSet: TPhonemeSet;
  AFileName: WideString);
begin
  SpeakToFile(WordToSSML(AWord, APhonemeSet));
end;

procedure Synthetize;
var
  LWord : TWord;
begin
  for LWord in Words do
    SpeakToFile(LWord, phIPA, LWord.Filenames.Audio+'.wav');
end;

//var i : integer;
//initialization
  //ListAvailableVoices;
  //Synthetize;
  //for i := 0 to High(Words) do begin
  //  Speak(Words[i], phIPA);
  //end;


end.

