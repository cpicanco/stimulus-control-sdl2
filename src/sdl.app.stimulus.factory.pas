{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.factory;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections
  , SDL2
  , sdl.app.stimulus.contract
  , sdl.app.stimulus
  , sdl.app.controls.custom
  ;

type

  TStimulusClass = class of TStimulus;

  TStimulusRegistry = specialize TDictionary<string, TStimulusClass>;

  TStimulusList = specialize TList<TStimulus>;

  { TStimulusFactory }

  TStimulusFactory = class sealed
    strict private
      class var Registry: TStimulusRegistry;
      class var StimulusList : TStimulusList;
      class constructor Create;
      class destructor Destroy;
    public
      class procedure Clear;
      class procedure RegisterStimulusClass(
        AStimulusKind: string; AStimulusClass: TStimulusClass); static;
      class function New(AOwner: TObject; AStimulusKind: string;
        ACallbacks : TCallbacks) : TStimulus; static;
  end;

implementation

uses sdl.app.stimulus.picture
   , sdl.app.stimulus.text
   , sdl.app.stimulus.audio
   , sdl.app.stimulus.speech
   ;

{ TStimulusFactory }

class constructor TStimulusFactory.Create;
begin
  Registry := TStimulusRegistry.Create;
  StimulusList := TStimulusList.Create;
end;

class destructor TStimulusFactory.Destroy;
begin
  Clear;
  StimulusList.Free;
  Registry.Free;
end;

class procedure TStimulusFactory.Clear;
var
  Stimulus : TStimulus;
begin
  for Stimulus in StimulusList do begin
    Stimulus.Free;
  end;
  StimulusList.Clear;
end;

class procedure TStimulusFactory.RegisterStimulusClass(AStimulusKind: string;
  AStimulusClass: TStimulusClass);
begin
  Registry.Add(AStimulusKind, AStimulusClass);
end;

class function TStimulusFactory.New(AOwner: TObject; AStimulusKind: string;
  ACallbacks: TCallbacks): TStimulus;
var
  StimulusClass : TStimulusClass;
begin
  if not Registry.TryGetValue(AStimulusKind, StimulusClass) then
    raise EArgumentException.CreateFmt(
      'Stimulus kind is not registered: %s %s', [AStimulusKind, StimulusClass]);
  StimulusList.Add(StimulusClass.Create);
  Result := StimulusList.Last;
  Result.Stimuli := AOwner;
  Result.OnResponse:=ACallbacks.OnResponse;
  Result.OnMouseDown:=ACallbacks.OnMouseDown;
  Result.OnMouseEnter:=ACallbacks.OnMouseEnter;
  Result.OnMouseExit:=ACallbacks.OnMouseExit;
  Result.OnMouseMove:=ACallbacks.OnMouseMove;
  Result.OnMouseUp:=ACallbacks.OnMouseUp;
  Result.OnNoResponse:=ACallbacks.OnNoResponse;
end;


initialization
  TStimulusFactory.RegisterStimulusClass('A', TAudioStimulus);
  TStimulusFactory.RegisterStimulusClass('B', TPictureStimulus);
  TStimulusFactory.RegisterStimulusClass('C', TTextStimulus);
  TStimulusFactory.RegisterStimulusClass('D', TSpeechStimulus);

end.
